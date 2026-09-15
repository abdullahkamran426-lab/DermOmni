from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import VisionAnalysisError

VISION_MAX_DIMENSION = 2048
VISION_JPEG_QUALITY = 90
# Smaller payload for free-tier / base64 chat-completion fallbacks (e.g. HF).
FALLBACK_MAX_DIMENSION = 1024
FALLBACK_JPEG_QUALITY = 85
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".3gpp"}


def get_video_mime_type(extension: str) -> str:
    """Map a video extension to a MIME type."""
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".3gpp": "video/3gpp",
    }
    return mime_types.get(extension.lower(), "video/mp4")


def detect_media_type(media_path: str | Path) -> str:
    """Return 'video' or 'image' based on file extension. Single source of truth."""
    return "video" if Path(media_path).suffix.lower() in VIDEO_EXTENSIONS else "image"


def _prepare_image_bytes(
    media_path: Path,
    *,
    max_dimension: int = VISION_MAX_DIMENSION,
    quality: int = VISION_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """Decode, EXIF-correct, alpha-flatten and JPEG-encode an image."""
    try:
        with Image.open(media_path) as source:
            source.verify()
        with Image.open(media_path) as source:
            has_alpha = source.mode in {"RGBA", "LA"} or "transparency" in source.info
            image = ImageOps.exif_transpose(source)
            if has_alpha:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=quality, optimize=True)
            return encoded.getvalue(), "image/jpeg"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise VisionAnalysisError(
            "The uploaded media could not be decoded for visual analysis.",
            code="invalid_media",
        ) from exc


def encode_image_to_data_url(
    image_path: str | Path,
    *,
    max_dimension: int = FALLBACK_MAX_DIMENSION,
    quality: int = FALLBACK_JPEG_QUALITY,
) -> str:
    """Convert an image to a base64 data: URL for chat-completion vision fallbacks."""
    img_path = Path(image_path)
    if detect_media_type(img_path) != "image":
        raise VisionAnalysisError(
            "Expected an image file but received a video file.",
            code="invalid_media_type",
        )
    try:
        image_bytes, _ = _prepare_image_bytes(img_path, max_dimension=max_dimension, quality=quality)
    except VisionAnalysisError:
        raise
    except Exception as exc:
        raise VisionAnalysisError(
            f"Could not process image for fallback analysis: {exc}",
            code="image_processing_error",
        ) from exc
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_vision_media(media_path: Path) -> tuple[bytes, str, str]:
    """Prepare image/video bytes for a multimodal provider.

    Single consolidated entry point — all Gemini/HF image+video paths
    should call this instead of re-implementing PIL/video logic.
    """
    media_path = Path(media_path)
    try:
        if detect_media_type(media_path) == "video":
            return (
                media_path.read_bytes(),
                get_video_mime_type(media_path.suffix.lower()),
                "video",
            )
        media_bytes, mime_type = _prepare_image_bytes(media_path)
        return media_bytes, mime_type, "image"
    except VisionAnalysisError:
        raise
    except (OSError, ValueError) as exc:
        raise VisionAnalysisError(
            "The uploaded media could not be read for visual analysis.",
            code="invalid_media",
        ) from exc


def prepare_vision_image(image_path: Path) -> tuple[bytes, str]:
    """Decode and normalize an image upload into a provider-safe JPEG payload."""
    media_bytes, mime_type, media_type = prepare_vision_media(Path(image_path))
    if media_type != "image":
        raise VisionAnalysisError(
            "Expected an image file but received a video file.",
            code="invalid_media_type",
        )
    return media_bytes, mime_type
