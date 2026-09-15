from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.tools import tool
from tavily import TavilyClient

# Import Hugging Face as fallback
try:
    from huggingface_vision import get_hf_analysis
except ImportError:
    get_hf_analysis = None

from common.cache import TTLCache, build_search_cache_key
from common.errors import ConfigurationError, SearchError, VisionAnalysisError
from common.gemini import (
    call_with_key_rotation,
    delete_remote_file,
    get_gemini_api_keys,
    needs_file_api,
    should_rotate_key,
    upload_file_and_wait,
)
from common.media import detect_media_type, prepare_vision_media

load_dotenv()

logger = logging.getLogger(__name__)

_TAVILY: TavilyClient | None = None
_TAVILY_SEARCH_TIMEOUT = float(os.environ.get("TAVILY_SEARCH_TIMEOUT", "15"))
_TAVILY_SEARCH_RETRIES = int(os.environ.get("TAVILY_SEARCH_RETRIES", "2"))
_TAVILY_RETRY_BACKOFF_SECONDS = float(os.environ.get("TAVILY_RETRY_BACKOFF_SECONDS", "0.5"))
_SEARCH_CACHE = TTLCache[str](
    ttl_seconds=float(os.environ.get("TAVILY_CACHE_TTL_SECONDS", "300")),
    max_entries=int(os.environ.get("TAVILY_CACHE_MAX_ENTRIES", "256")),
)



def _tavily_client() -> TavilyClient:
    global _TAVILY
    if _TAVILY is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Missing TAVILY_API_KEY in the environment.",
                code="missing_api_key",
            )
        _TAVILY = TavilyClient(api_key=api_key)
    return _TAVILY


def _vision_models() -> list[str]:
    """Return the primary + fallback vision model names, deduplicated in order."""
    primary = os.environ.get("GEMINI_VISION_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3.6-flash"
    fallback = os.environ.get("GEMINI_VISION_FALLBACK_MODEL", "gemini-3.6-flash")

    models = []
    for candidate in (primary, fallback):
        name = candidate.strip()
        if name and name not in models:
            models.append(name)
    return models


def _can_retry_with_fallback(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ("not found", "unsupported model", "invalid model", "does not support"))


def _vision_config() -> types.GenerateContentConfig:
    # Keep the production-verified thinking configuration unchanged.
    return types.GenerateContentConfig(
        system_instruction=(
            "You are a clinical dermatology research assistant performing "
            "educational visual analysis of skin images and videos. Be precise, clinical, "
            "and structured. Never provide an official diagnosis. "
            "For videos, analyze the visual content, movement, and any visible changes over time. "
            "Output plain text only, with no markdown or special characters. "
            "The answer should be readable by a patient with no medical background."
        ),
        temperature=0.7,
        max_output_tokens=1600,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        response_mime_type="text/plain",
    )


def _generate_vision_response(client: genai.Client, media_bytes: bytes, mime_type: str, prompt: str, model: str):
    return client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=media_bytes, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ],
        config=_vision_config(),
    )


def _generate_vision_file_response(
    client: genai.Client, media_path: Path, prompt: str, model: str
):
    """File API path for videos / >10MB files (streams from disk, no OOM)."""
    uploaded = upload_file_and_wait(client, media_path)
    try:
        return client.models.generate_content(
            model=model,
            contents=[uploaded, types.Part.from_text(text=prompt)],
            config=_vision_config(),
        )
    finally:
        delete_remote_file(client, uploaded)


def _run_vision_analysis(media_path: Path, prompt: str, patient_query: str = "") -> str:
    media_path = Path(media_path)
    try:
        media_type = detect_media_type(media_path)
    except Exception:
        media_type = "image"
    # Fail fast when no keys configured (500, not 502).
    get_gemini_api_keys()

    # Small images stay inline (single PIL prep); videos / >10MB use File API.
    use_file_api = needs_file_api(media_path)
    media_bytes: bytes | None = None
    mime_type = ""
    if not use_file_api:
        try:
            media_bytes, mime_type, prepared_type = prepare_vision_media(media_path)
            media_type = prepared_type
        except VisionAnalysisError:
            raise
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.error("Gemini vision client initialization failed: %s", exc)
            raise VisionAnalysisError(
                "Gemini vision is not configured correctly.", code="vision_client_error"
            ) from exc

    def _attempt_with_rotation(model: str):
        def _op(client: genai.Client):
            if use_file_api:
                return _generate_vision_file_response(client, media_path, prompt, model)
            assert media_bytes is not None
            return _generate_vision_response(client, media_bytes, mime_type, prompt, model)

        return call_with_key_rotation(
            _op,
            lambda key: genai.Client(api_key=key),
            operation_name=f"research-vision-{model}",
        )

    last_error: Exception | None = None
    response: Any = None
    model_used: str | None = None
    models = _vision_models()
    for index, model in enumerate(models):
        try:
            response = _attempt_with_rotation(model)
            model_used = model
            logger.info("Gemini vision analysis completed with model %s for %s", model, media_type)
            break
        except (VisionAnalysisError, ConfigurationError):
            raise
        except Exception as exc:
            last_error = exc
            if should_rotate_key(exc):
                # Key rotation already exhausted inside the helper (quota or
                # bad-key); fall through to HF fallback below.
                logger.warning("Gemini vision key failed (model %s), trying HF fallback: %s", model, exc)
                break
            logger.warning("Gemini vision model %s failed: %s", model, exc)
            if index == 0 and len(models) > 1 and _can_retry_with_fallback(exc):
                continue
            break

    if response is None:
        if get_hf_analysis is not None:
            logger.info("Gemini failed, attempting Hugging Face fallback for %s analysis", media_type)
            try:
                return get_hf_analysis(media_path, media_type, patient_query)
            except Exception as hf_exc:
                logger.warning("Hugging Face fallback also failed: %s", hf_exc)
                last_error = hf_exc

        if last_error and _can_retry_with_fallback(last_error):
            raise VisionAnalysisError("The configured Gemini vision model is unavailable.", code="vision_model_unavailable") from last_error
        raise VisionAnalysisError(
            f"Vision analysis failed while processing the {media_type}. Both Gemini and Hugging Face unavailable.",
            code="vision_request_failed",
        ) from last_error

    if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
        usage = response.usage_metadata
        logger.warning(
            "Gemini vision response truncated by MAX_TOKENS (model=%s, thoughts_tokens=%s, output_tokens=%s)",
            model_used,
            getattr(usage, "thoughts_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )

    try:
        text = response.text or ""
    except ValueError as exc:
        raise VisionAnalysisError("Gemini returned no usable visual analysis.", code="empty_response") from exc
    if not text.strip():
        raise VisionAnalysisError("Gemini returned no usable visual analysis.", code="empty_response")
    return text.strip()


def analyze_skin_image(image_path: str | Path, patient_query: str = "") -> str:
    """Use Gemini multimodal to visually inspect a skin image."""
    img_path = Path(image_path)
    prompt = (
        "Analyze this skin image as a clinical dermatology research assistant. "
        "Structure your response in two parts:\n\n"
        "VISUAL CHARACTERISTICS: Describe what you observe — colour, texture, "
        "distribution pattern, border definition, surface features (scaling, "
        "crusting, vesicles, papules, plaques, etc.), and the approximate body "
        "area affected.\n\n"
        "CONDITION CANDIDATES: List the top 3 most likely skin conditions based "
        "solely on the visual evidence, ranked by likelihood, with a one-sentence "
        "rationale for each. Do not provide an official diagnosis — this is an "
        "educational visual assessment only."
    )
    if patient_query:
        prompt += f"\n\nPatient's reported concern: {patient_query}"
    return _run_vision_analysis(img_path, prompt, patient_query)


def analyze_skin_video(video_path: str | Path, patient_query: str = "") -> str:
    """Use Gemini multimodal to visually inspect a skin video."""
    vid_path = Path(video_path)
    prompt = (
        "Analyze this skin video as a clinical dermatology research assistant. "
        "Structure your response in three parts:\n\n"
        "VISUAL CHARACTERISTICS OVER TIME: Describe what you observe throughout the video — "
        "colour changes, texture variations, movement patterns, distribution changes, "
        "border definition, surface features (scaling, crusting, vesicles, papules, plaques, etc.), "
        "and the approximate body area affected. Note any progression or regression.\n\n"
        "DYNAMIC FEATURES: Describe any movement, response to touch or stimuli, changes in "
        "appearance over time, or other dynamic characteristics visible in the video.\n\n"
        "CONDITION CANDIDATES: List the top 3 most likely skin conditions based "
        "on the visual evidence and temporal patterns, ranked by likelihood, with a one-sentence "
        "rationale for each. Do not provide an official diagnosis — this is an "
        "educational visual assessment only."
    )
    if patient_query:
        prompt += f"\n\nPatient's reported concern: {patient_query}"
    return _run_vision_analysis(vid_path, prompt, patient_query)


MEDICAL_DOMAINS = [
    "ncbi.nlm.nih.gov",
    "aad.org",
    "dermnetnz.org",
    "mayoclinic.org",
    "healthline.com",
    "webmd.com",
    "nih.gov",
    "clevelandclinic.org",
]
PAPER_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "jamanetwork.com",
    "nejm.org",
    "springer.com",
    "sciencedirect.com",
    "bmj.com",
]


def _is_retryable_search_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("429", "500", "502", "503", "504", "rate limit", "temporarily unavailable", "timeout", "connection reset"))


def _retry_tavily_search(
    query: str,
    *,
    operation: str,
    domains: list[str],
    max_results: int,
) -> dict[str, Any]:
    client = _tavily_client()
    cache_key = build_search_cache_key(
        operation=operation,
        query=query,
        model="tavily-search",
        search_depth="advanced",
        max_results=max_results,
        domains=domains,
    )
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("Tavily cache hit: %s", operation)
        return cached

    attempts = max(1, _TAVILY_SEARCH_RETRIES + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=domains,
                timeout=_TAVILY_SEARCH_TIMEOUT,
            )
            if not isinstance(result, dict):
                raise SearchError("Tavily returned an unexpected response shape.", code="invalid_search_response")
            _SEARCH_CACHE.set(cache_key, result)
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1 or not _is_retryable_search_error(exc):
                break
            delay = _TAVILY_RETRY_BACKOFF_SECONDS * (2**attempt)
            logger.warning("Tavily %s failed (attempt %d/%d); retrying in %.2fs: %s", operation, attempt + 1, attempts, delay, exc)
            time.sleep(delay)

    raise SearchError(f"Tavily {operation} failed after {attempts} attempt(s).", code="search_request_failed") from last_error


def _format_search_results(results: dict[str, Any], *, paper: bool) -> str:
    prefix = "Paper" if paper else "Title"
    body_key = "Abstract" if paper else "Snippet"
    limit = 500 if paper else 400
    out: list[str] = []
    for result in results.get("results", []):
        title = result.get("title")
        url = result.get("url")
        content = result.get("content", "")
        if not title or not url:
            continue
        out.append(f"{prefix}: {title}\nURL: {url}\n{body_key}: {content[:limit]}\n")
    if not out:
        return "No peer-reviewed papers found." if paper else "No relevant medical sources found."
    return "\n----\n".join(out)


@tool
def medical_web_search(query: str) -> str:
    """Search trusted dermatology and medical websites."""
    results = _retry_tavily_search(query, operation="medical_web", domains=MEDICAL_DOMAINS, max_results=6)
    return _format_search_results(results, paper=False)


@tool
def research_paper_search(query: str) -> str:
    """Search PubMed and peer-reviewed journals for clinical studies on skin conditions."""
    search_query = f"{query} clinical study dermatology peer-reviewed"
    results = _retry_tavily_search(search_query, operation="research_papers", domains=PAPER_DOMAINS, max_results=5)
    return _format_search_results(results, paper=True)
