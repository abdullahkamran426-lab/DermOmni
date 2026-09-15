"""
Hugging Face Vision Analysis Module
Free fallback for IMAGE analysis when Gemini is unavailable, using Hugging
Face's Inference Providers chat-completion API (OpenAI-compatible).

Video is intentionally NOT supported here — see analyze_video_with_hf().
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from common.errors import ConfigurationError
from common.media import encode_image_to_data_url

load_dotenv()

logger = logging.getLogger(__name__)

# A vision-language chat model with broad availability across HF Inference
# Providers. Override with HF_VISION_MODEL if you prefer a different one
# (e.g. "zai-org/GLM-4.5V" or "meta-llama/Llama-3.2-11B-Vision-Instruct").
# ":cheapest" suffix auto-routes across your enabled providers.
DEFAULT_HF_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct:cheapest"

# Timeout (in seconds) for a single Hugging Face inference request.
_HF_TIMEOUT = 30  # seconds


class HuggingFaceVisionError(RuntimeError):
    """Custom error for Hugging Face vision analysis failures."""

    def __init__(self, message: str, *, code: str = "hf_vision_error") -> None:
        super().__init__(message)
        self.code = code


def _hf_api_key() -> str:
    """Get Hugging Face API key from environment."""
    api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "Missing HF_API_KEY or HUGGINGFACE_API_KEY in the environment.",
            code="missing_api_key",
        )
    return api_key


def _encode_image_to_data_url(image_path: Path) -> str:
    """Convert an image to a base64 data: URL (delegates to common.media)."""
    from common.errors import VisionAnalysisError

    try:
        return encode_image_to_data_url(image_path)
    except VisionAnalysisError as exc:
        raise HuggingFaceVisionError(
            f"Could not process image for Hugging Face analysis: {exc}",
            code=getattr(exc, "code", "image_processing_error"),
        ) from exc


def analyze_image_with_hf(image_path: str | Path, patient_query: str = "") -> str:
    """
    Analyze a skin image via a Hugging Face vision-language chat model.

    Uses the Inference Providers chat-completion API (the current, supported
    replacement for the deprecated api-inference.huggingface.co endpoint).

    Raises:
        HuggingFaceVisionError: If the image can't be processed, the request
                                 fails, or the response is empty/malformed.
        RuntimeError: If HF_API_KEY / HUGGINGFACE_API_KEY is missing.
    """
    img_path = Path(image_path)
    data_url = _encode_image_to_data_url(img_path)

    prompt = (
        "Describe this skin condition in clinical detail: colour, texture, "
        "distribution, border definition, and any visible surface patterns "
        "(scaling, crusting, vesicles, papules, plaques). This is for "
        "educational purposes only — do not provide an official diagnosis."
    )
    if patient_query:
        prompt += f" Patient reports: {patient_query}"

    client = InferenceClient(
        provider=os.environ.get("HF_INFERENCE_PROVIDER", "auto"),
        api_key=_hf_api_key(),
        timeout=_HF_TIMEOUT,
    )
    model = os.environ.get("HF_VISION_MODEL", DEFAULT_HF_VISION_MODEL)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=400,
        )
    except Exception as exc:
        logger.error("Hugging Face vision request failed: %s", exc)
        raise HuggingFaceVisionError(
            f"Hugging Face vision request failed: {exc}",
            code="api_request_error",
        ) from exc

    try:
        text = completion.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise HuggingFaceVisionError(
            "Hugging Face returned an unexpected response shape.",
            code="bad_response_shape",
        ) from exc

    if not text or not text.strip():
        raise HuggingFaceVisionError(
            "Hugging Face returned an empty response.",
            code="empty_response",
        )
    return text.strip()


def analyze_video_with_hf(video_path: str | Path, patient_query: str = "") -> str:
    """
    Video analysis is not supported by this fallback.

    There is no consistently-available free serverless video-understanding
    model across Hugging Face's Inference Providers today, and this project
    deliberately avoids an ffmpeg/OpenCV dependency just for frame
    extraction (see README). Callers should catch HuggingFaceVisionError
    here and degrade to text-only analysis, which the research pipeline
    already does.
    """
    raise HuggingFaceVisionError(
        "The Hugging Face fallback does not support video analysis. "
        "Falling back to the patient's written description instead.",
        code="video_not_supported",
    )


def get_hf_analysis(
    media_path: str | Path,
    media_type: str = "auto",
    patient_query: str = "",
) -> str:
    """
    Analyze media (image or video) using Hugging Face.

    Video always raises HuggingFaceVisionError(code="video_not_supported") —
    see analyze_video_with_hf() for why.
    """
    path = Path(media_path)

    if media_type == "auto":
        video_extensions = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".3gpp"}
        media_type = "video" if path.suffix.lower() in video_extensions else "image"

    if media_type == "video":
        return analyze_video_with_hf(path, patient_query)
    return analyze_image_with_hf(path, patient_query)