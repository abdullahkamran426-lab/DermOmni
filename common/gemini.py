"""Shared Gemini helpers: multi-key rotation + File API upload.

Single source of truth for:
- GEMINI_API_KEY (primary) + GEMINI_API_KEYS (comma/newline-separated backups).
- Failover detection (429 quota AND invalid-key/auth errors) → try next key.
- File API upload with state polling for videos / files >10MB.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .errors import ConfigurationError
from .media import detect_media_type

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Files larger than this use File API instead of inline bytes (prevents OOM).
FILE_API_THRESHOLD_BYTES = int(os.environ.get("GEMINI_FILE_API_THRESHOLD_BYTES", str(10 * 1024 * 1024)))
FILE_API_POLL_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_FILE_POLL_TIMEOUT", "60"))
FILE_API_POLL_INTERVAL_SECONDS = float(os.environ.get("GEMINI_FILE_POLL_INTERVAL", "2"))


def get_gemini_api_keys() -> list[str]:
    """Parse all configured Gemini keys. GEMINI_API_KEY first, then GEMINI_API_KEYS list."""
    candidates: list[str] = []
    # Primary key first so "first API errors → second API activates".
    primary = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if primary:
        candidates.append(primary)
    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    # Accept comma, newline, or semicolon separators.
    for chunk in raw_multi.replace(";", ",").replace("\n", ",").split(","):
        key = chunk.strip().strip('"').strip("'")
        if key:
            candidates.append(key)
    # Dedupe preserving order.
    seen: dict[str, None] = {}
    for key in candidates:
        seen.setdefault(key, None)
    keys = list(seen.keys())
    if not keys:
        raise ConfigurationError(
            "Missing GEMINI_API_KEYS (or GEMINI_API_KEY) in the environment.",
            code="missing_api_key",
        )
    return keys


def is_quota_error(exc: BaseException) -> bool:
    """True when the error looks like a per-key rate limit / quota exhaustion."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    code = getattr(exc, "code", None)
    try:
        if code is not None and int(code) == 429:
            return True
    except (TypeError, ValueError):
        pass
    message = str(exc).lower()
    markers = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "ratelimit",
        "too many requests",
    )
    return any(marker in message for marker in markers)


def is_invalid_key_error(exc: BaseException) -> bool:
    """True when the error looks like a bad/revoked/unauthorized API key."""
    status = getattr(exc, "status_code", None)
    if status in {400, 401, 403}:
        message = str(exc).lower()
        # Only treat 400/401/403 as key errors when the message points at the key.
        # (Prevents rotating keys on unrelated bad-request / permission errors.)
        key_markers = (
            "api key",
            "apikey",
            "api_key",
            "invalid key",
            "key not valid",
            "key expired",
            "unauthenticated",
            "unauthorized",
            "permission denied",
            "leaked",
            "blocked",
        )
        if any(marker in message for marker in key_markers):
            return True
        # google-genai ApiError often carries code=400/401/403 with a short
        # message; status alone is still a strong key-error signal, so check
        # the numeric code below as well.
    code = getattr(exc, "code", None)
    try:
        if code is not None and int(code) in {400, 401, 403}:
            message = str(exc).lower()
            if any(
                marker in message
                for marker in (
                    "api key",
                    "apikey",
                    "api_key",
                    "invalid key",
                    "unauthenticated",
                    "unauthorized",
                    "permission denied",
                )
            ):
                return True
    except (TypeError, ValueError):
        pass
    message = str(exc).lower()
    markers = (
        "api_key_invalid",
        "api key not valid",
        "invalid api key",
        "invalid key",
        "key expired",
        "api key expired",
        "unauthenticated",
        "permission denied",
    )
    return any(marker in message for marker in markers)


def should_rotate_key(exc: BaseException) -> bool:
    """True when trying the next API key could help (quota OR bad-key errors)."""
    return is_quota_error(exc) or is_invalid_key_error(exc)


def is_transient_error(exc: BaseException) -> bool:
    """True for temporary 503/UNAVAILABLE/overloaded errors worth retrying."""
    status = getattr(exc, "status_code", None)
    if status in {503, 500, 502, 504}:
        return True
    code = getattr(exc, "code", None)
    try:
        if code is not None and int(code) in {500, 502, 503, 504}:
            return True
    except (TypeError, ValueError):
        pass
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "503",
            "unavailable",
            "high demand",
            "overloaded",
            "temporarily",
            "try again later",
        )
    )


def call_with_key_rotation(
    operation: Callable[[Any], T],
    client_factory: Callable[[str], Any],
    *,
    operation_name: str = "gemini_request",
    max_transient_retries: int = 2,
) -> T:
    """Run `operation(client)` trying each API key in turn on key errors.

    Quota errors (429) AND invalid-key/auth errors (400/401/403 API_KEY_INVALID,
    unauthenticated, permission denied) rotate to the next key. Transient
    503/UNAVAILABLE errors are retried on the SAME key with backoff (1s, 2s)
    before propagating. Any other exception propagates immediately. If every
    key fails, the last error propagates so callers can map it to
    GuidanceError / VisionAnalysisError.
    """
    keys = get_gemini_api_keys()
    last_error: BaseException | None = None
    for index, key in enumerate(keys):
        client = client_factory(key)
        try:
            return operation(client)
        except Exception as exc:
            if should_rotate_key(exc) and index < len(keys) - 1:
                reason = "quota" if is_quota_error(exc) else "invalid-key"
                logger.warning(
                    "%s hit %s error on key %d/%d, rotating: %s",
                    operation_name, reason, index + 1, len(keys), exc,
                )
                last_error = exc
                continue
            if is_transient_error(exc):
                for retry in range(1, max_transient_retries + 1):
                    delay = 1.0 * retry
                    logger.warning(
                        "%s transient 503 (attempt %d/%d), retrying in %.1fs: %s",
                        operation_name, retry, max_transient_retries, delay, exc,
                    )
                    time.sleep(delay)
                    try:
                        return operation(client)
                    except Exception as retry_exc:
                        exc = retry_exc
                        if should_rotate_key(retry_exc) and index < len(keys) - 1:
                            break
                        if not is_transient_error(retry_exc):
                            raise
                if should_rotate_key(exc) and index < len(keys) - 1:
                    last_error = exc
                    continue
            raise
    # Unreachable in practice (loop either returns or raises), kept for typing.
    assert last_error is not None
    raise last_error


# Alias kept for the QA/test-suite naming convention.
execute_with_key_rotation = call_with_key_rotation


def needs_file_api(media_path: str | Path) -> bool:
    """Videos always use File API; images only when over the byte threshold."""
    path = Path(media_path)
    if detect_media_type(path) == "video":
        return True
    try:
        return path.stat().st_size > FILE_API_THRESHOLD_BYTES
    except OSError:
        return False


def upload_file_and_wait(client: Any, media_path: Path, *, mime_type: str | None = None) -> Any:
    """Upload via client.files.upload and poll until ACTIVE (or raise)."""
    path = Path(media_path)
    if not path.is_file():
        from .errors import VisionAnalysisError

        raise VisionAnalysisError("Media file not found for upload.", code="invalid_media")
    logger.info("Uploading %s (%d bytes) via Gemini File API", path.name, path.stat().st_size)
    try:
        uploaded = client.files.upload(file=str(path))
    except TypeError:
        # Older SDK signature: upload(path=...) / file kwarg variations.
        uploaded = client.files.upload(path=str(path))

    name = getattr(uploaded, "name", None)
    deadline = time.monotonic() + FILE_API_POLL_TIMEOUT_SECONDS
    current = uploaded
    while time.monotonic() < deadline:
        state = getattr(current, "state", None)
        # No explicit state (some SDK versions) → assume ready after upload.
        if state is None:
            return current
        state_name = getattr(state, "name", state)
        if isinstance(state_name, str):
            upper = state_name.upper()
            if upper in {"ACTIVE", "READY"}:
                return current
            if upper in {"FAILED", "ERROR", "DELETED"}:
                from .errors import VisionAnalysisError

                raise VisionAnalysisError(
                    f"Gemini file processing failed (state={state_name}).",
                    code="vision_request_failed",
                )
        time.sleep(FILE_API_POLL_INTERVAL_SECONDS)
        try:
            if name:
                current = client.files.get(name=name)
            else:
                break
        except Exception as exc:
            logger.warning("File poll failed for %s: %s", name, exc)
            break
    return current


def delete_remote_file(client: Any, uploaded: Any) -> None:
    """Best-effort cleanup of a remote Gemini File API object."""
    try:
        name = getattr(uploaded, "name", None)
        if name and hasattr(client, "files"):
            client.files.delete(name=name)
            logger.info("Deleted remote Gemini file %s", name)
    except Exception as exc:
        logger.warning("Could not delete remote Gemini file: %s", exc)
