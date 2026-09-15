from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx
from deepgram import DeepgramClient
from deepgram.core.api_error import ApiError
from dotenv import load_dotenv

from common.env import read_float_env, read_int_env
from common.errors import ConfigurationError, TranscriptionError

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "nova-3"
MAX_INLINE_AUDIO_BYTES = 20 * 1024 * 1024

# HTTP status codes worth retrying: rate limits and transient server errors.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# Shared env-var parsers live in common.env (single source of truth).
DEEPGRAM_TIMEOUT_SECONDS = read_float_env("DEEPGRAM_TIMEOUT_SECONDS", 45.0)
DEEPGRAM_STT_MAX_ATTEMPTS = read_int_env("DEEPGRAM_STT_MAX_ATTEMPTS", 2)


def transcribe_patient_voice(audio_filepath: str | Path) -> str:
    """
    Transcribe a patient's spoken audio to text via Deepgram.

    Retries on transport failures and retryable HTTP statuses (429, 5xx).
    Client errors (4xx other than 429) and empty-transcript results fail
    immediately — retrying the same audio bytes won't change the outcome.

    Raises:
        ValueError:   If the file is missing, unreadable, or too large.
        ConfigurationError: If DEEPGRAM_API_KEY is missing.
        TranscriptionError: If the provider returns no speech text (code=no_speech)
                      or all attempts fail (code=stt_unavailable / stt_client_error).
    """
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise ConfigurationError(
            "Missing DEEPGRAM_API_KEY in the environment.",
            code="missing_api_key",
        )

    audio_path = Path(audio_filepath)
    if not audio_path.is_file():
        raise ValueError("The uploaded audio file could not be found.")
    if audio_path.stat().st_size > MAX_INLINE_AUDIO_BYTES:
        raise ValueError("The uploaded audio file is too large for transcription.")

    try:
        audio_bytes = audio_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read the audio file: {exc}") from exc

    # Stateless client — safe to build once outside the retry loop.
    deepgram = DeepgramClient(
        api_key=deepgram_api_key,
        timeout=DEEPGRAM_TIMEOUT_SECONDS,
        max_retries=0,  # We control retries here.
    )

    last_error: Exception | None = None
    for attempt in range(1, DEEPGRAM_STT_MAX_ATTEMPTS + 1):
        try:
            transcription = deepgram.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=os.environ.get("DEEPGRAM_STT_MODEL", DEFAULT_STT_MODEL),
                language="en",
                punctuate=True,
                smart_format=True,
            )

            try:
                text = transcription.results.channels[0].alternatives[0].transcript.strip()
            except (AttributeError, IndexError, TypeError) as exc:
                raise TranscriptionError(
                    "The transcription provider returned no speech text.",
                    code="no_speech",
                ) from exc

            if not text:
                raise TranscriptionError(
                    "No speech could be transcribed from the audio.",
                    code="no_speech",
                )

            logger.info("Transcribed patient audio (%d characters)", len(text))
            return text

        except ApiError as exc:
            if exc.status_code not in _RETRYABLE_STATUS_CODES:
                raise TranscriptionError(
                    f"Deepgram STT failed with a non-retryable status {exc.status_code}.",
                    code="stt_client_error",
                ) from exc
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        except TranscriptionError:
            # Empty transcript / malformed payload — retrying same bytes won't help.
            raise

        logger.warning(
            "Deepgram STT attempt %s/%s failed: %s",
            attempt, DEEPGRAM_STT_MAX_ATTEMPTS, last_error,
        )
        if attempt < DEEPGRAM_STT_MAX_ATTEMPTS:
            time.sleep(attempt)

    raise TranscriptionError(
        "Deepgram transcription was unavailable after retrying.",
        code="stt_unavailable",
    ) from last_error