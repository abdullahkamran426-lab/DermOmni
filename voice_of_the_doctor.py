from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.api_error import ApiError

from common.env import read_float_env, read_int_env
from common.errors import ConfigurationError, SpeechSynthesisError

logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"
MAX_TTS_CHARACTERS = 2_000

# HTTP status codes worth retrying: rate limits and transient server errors.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# Shared env-var parsers live in common.env (single source of truth).
DEEPGRAM_TIMEOUT_SECONDS = read_float_env("DEEPGRAM_TIMEOUT_SECONDS", 45.0)
DEEPGRAM_MAX_ATTEMPTS = read_int_env("DEEPGRAM_MAX_ATTEMPTS", 2)


def _text_for_speech(text: str) -> str:
    """Keep speech input within Deepgram's 2,000-character request limit."""
    text = " ".join(text.split())
    if len(text) <= MAX_TTS_CHARACTERS:
        return text

    candidate = text[:MAX_TTS_CHARACTERS]
    sentence_end = max(candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? "))
    if sentence_end >= 1_000:
        return candidate[: sentence_end + 1].strip()
    return candidate[: MAX_TTS_CHARACTERS - 3].rstrip() + "..."


def convert_text_to_doctor_audio(
    text: str,
    output_filepath: str | Path = DEFAULT_DOCTOR_AUDIO,
) -> Path:
    """
    Synthesize `text` to speech via Deepgram and save it as an MP3.

    Retries on transport failures and retryable HTTP statuses (429, 5xx).
    Client errors (4xx other than 429) fail immediately — retrying won't help
    a bad API key or malformed request.

    Raises:
        ValueError:   If text is empty.
        ConfigurationError: If DEEPGRAM_API_KEY is missing.
        SpeechSynthesisError: If synthesis fails (codes: tts_unavailable,
            tts_client_error, tts_empty_response).
    """
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise ConfigurationError(
            "Missing DEEPGRAM_API_KEY in the environment.",
            code="missing_api_key",
        )
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    speech_text = _text_for_speech(text)

    # Stateless client — safe to build once outside the retry loop.
    deepgram = DeepgramClient(
        api_key=deepgram_api_key,
        timeout=DEEPGRAM_TIMEOUT_SECONDS,
        max_retries=0,  # We control retries here so the file is only
                        # replaced once the full response is buffered.
    )

    last_error: Exception | None = None
    for attempt in range(1, DEEPGRAM_MAX_ATTEMPTS + 1):
        try:
            audio = deepgram.speak.v1.audio.generate(
                text=speech_text,
                model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
                encoding="mp3",
            )
            audio_bytes = b"".join(audio)
            if not audio_bytes:
                raise SpeechSynthesisError(
                    "Deepgram returned an empty audio response.",
                    code="tts_empty_response",
                )

            output_filepath.write_bytes(audio_bytes)
            logger.info("Saved doctor audio to %s (%d bytes)", output_filepath, len(audio_bytes))
            return output_filepath

        except ApiError as exc:
            if exc.status_code not in _RETRYABLE_STATUS_CODES:
                raise SpeechSynthesisError(
                    f"Deepgram TTS failed with a non-retryable status {exc.status_code}.",
                    code="tts_client_error",
                ) from exc
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        except SpeechSynthesisError:
            raise

        logger.warning(
            "Deepgram TTS attempt %s/%s failed: %s",
            attempt, DEEPGRAM_MAX_ATTEMPTS, last_error,
        )
        if attempt < DEEPGRAM_MAX_ATTEMPTS:
            time.sleep(attempt)

    raise SpeechSynthesisError(
        "Deepgram TTS was unavailable after retrying.",
        code="tts_unavailable",
    ) from last_error