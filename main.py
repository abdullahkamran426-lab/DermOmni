from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from typing import Any

from brain_of_the_doctor_gemini import brain_of_the_doctor, clean_doctor_response
from voice_of_the_doctor import convert_text_to_doctor_audio
from voice_of_the_patient import transcribe_patient_voice
from Skin_research_pipeline import run_skin_research_pipeline
from common import history_store
from common.localization import parse_accept_language
from common.image_annotator import annotate_image
from common.dermatology_rag import get_dermatology_rag
from common.gemini import call_with_key_rotation, get_gemini_api_keys, is_quota_error
from common.history_store import sanitize_user_id
from common.pdf_report import build_consultation_pdf
from common.errors import (
    ConfigurationError,
    GuidanceError,
    SpeechSynthesisError,
    TranscriptionError,
    VisionAnalysisError,
)
from google import genai
from google.genai import types


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = ROOT_DIR / "frontend" / "redesigned.html"
AUDIO_DIR = ROOT_DIR / "generated_audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Consultation-history / feedback store (SQLite, WAL). Created lazily so
# importing the app in tests never fails on a read-only FS.
# NOTE: logging must be configured BEFORE this try/except (P0 fix) —
# previously `logger` was defined ~180 lines below, so an OSError here
# raised NameError instead of a graceful warning.
try:
    history_store.init_db()
except OSError:
    logger.warning("History store unavailable; history endpoints will 503.")


class FeedbackIn(BaseModel):
    consultation_id: str = Field(min_length=1)
    rating: str = Field(min_length=1, description="'up' or 'down' (also accepts 1/-1)")
    note: str = Field(default="", max_length=2000)
    user_id: str = Field(default="")


class ChatTurn(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    consultation_id: str | None = Field(default=None)
    user_id: str = Field(default="")
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


def _resolve_user_id(request: Request, explicit: str = "") -> str:
    """Per-user context: explicit field > X-User-Id header > 'anonymous'."""
    candidate = (explicit or "").strip() or request.headers.get("X-User-Id", "")
    return sanitize_user_id(candidate)


def _media_kind(*, has_audio=False, has_image=False, has_video=False) -> str:
    """Describe which upload types were provided (for history metadata)."""
    active_kinds = []

    if has_audio:
        active_kinds.append("audio")
    if has_image:
        active_kinds.append("image")
    if has_video:
        active_kinds.append("video")

    if len(active_kinds) == 0:
        return "none"
    if len(active_kinds) > 1:
        return "mixed"
    return active_kinds[0]


def _model_meta() -> dict:
    return {
        "consult_model": os.getenv("GEMINI_CONSULT_MODEL", "gemini-3.6-flash"),
        "stt_model": "deepgram:nova-3",
        "tts_model": "deepgram:aura-2-thalia-en",
    }


_CHAT_SYSTEM = (
    "You are a follow-up assistant for a skin-care consultation app. "
    "Answer the patient's follow-up question using the stored session summary "
    "and the recent conversation turns. Stay educational: never state a "
    "definitive diagnosis, never name or prescribe medications, and always "
    "recommend an in-person dermatologist for anything painful, spreading, "
    "worsening, or persistent. If the question goes beyond the stored session, "
    "answer generally and suggest starting a new consultation with a fresh "
    "photo or description. Output plain text only — no markdown, bullets, or "
    "emojis. Keep the reply concise (4-6 sentences)."
)


def _chat_client_for_key(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _run_chat_reply(prompt: str) -> str:
    """Text-only Gemini follow-up reply with key rotation (patchable in tests)."""
    from brain_of_the_doctor_gemini import DEFAULT_GEMINI_MODEL

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    get_gemini_api_keys()  # fail fast with ConfigurationError when unconfigured

    def _op(client: genai.Client):
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_CHAT_SYSTEM,
                temperature=0.7,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
                response_mime_type="text/plain",
            ),
        )

    try:
        response = call_with_key_rotation(_op, _chat_client_for_key, operation_name="chat-followup")
    except ConfigurationError:
        raise
    except Exception as exc:
        if is_quota_error(exc):
            raise GuidanceError(
                "The guidance provider is rate-limited. Please try again shortly.",
                code="quota_exhausted",
            ) from exc
        raise GuidanceError("The guidance provider request failed.", code="guidance_request_failed") from exc
    try:
        text = response.text
    except ValueError as exc:
        raise GuidanceError("The provider returned no usable text.", code="empty_response") from exc
    cleaned = clean_doctor_response(text or "")
    if not cleaned.strip():
        raise GuidanceError("The provider returned an empty response.", code="empty_response")
    return cleaned


def _build_chat_prompt(
    message: str, record: dict | None, turns: list[dict]
) -> str:
    """Ground a follow-up question in the stored session + recent turns."""
    lines: list[str] = []
    if record is not None:
        kind = record.get("kind", "consult")
        main_text = (record.get("report") or record.get("guidance") or "")[:3000]
        visual = (record.get("visual_analysis") or "")[:1500]
        patient = (record.get("input_text") or record.get("transcript") or "")[:1500]
        evidence = (record.get("evidence") or {}).get("evidence_level", "")
        lines.append(f"Stored session ({kind}, {record.get('created_at', '')}):")
        if patient:
            lines.append(f"Patient description: {patient}")
        if main_text:
            lines.append(f"Prior guidance/report: {main_text}")
        if visual:
            lines.append(f"Visual analysis: {visual}")
        if evidence:
            lines.append(f"Evidence level: {evidence}")
    else:
        lines.append("No stored session selected; answer generally.")
    for turn in turns[-8:]:
        role = "Patient" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(turn.get('content') or '')[:1000]}")
    lines.append(f"New patient question: {message.strip()}")
    return "\n".join(lines)


# Startup validation
if not FRONTEND_FILE.exists():
    raise RuntimeError(
        f"Frontend file not found at {FRONTEND_FILE}. "
        "Ensure 'frontend/code.html' exists in the project root."
    )

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_IMAGE_DIMENSION = 10_000

# How long a generated response mp3 is kept before being cleaned up.
AUDIO_RETENTION_SECONDS = 60 * 60  # 1 hour

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".jfif"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".3gpp"}
CONTENT_TYPE_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/x-flv": ".flv",
    "video/x-ms-wmv": ".wmv",
    "video/3gpp": ".3gpp",
}

limiter = Limiter(key_func=get_remote_address)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded (5/min). Please try again shortly."},
    )


app = FastAPI(title="DermOmni")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    # "null" intentionally excluded: sandboxed iframes and file:// pages send
    # Origin: null, which would otherwise bypass this allow-list entirely.
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the repository's browser frontend."""
    return FileResponse(FRONTEND_FILE, media_type="text/html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/api/research")
async def research(
    request: Request,
    query: str = Form(""),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    user_id: str = Form(""),
) -> dict:
    """
    Agent Research System endpoint.

    Accepts a text query plus an optional image or video, runs the full
    skin research pipeline, and returns a structured JSON report.

    Video analysis is fully supported - Gemini will analyze video content
    including temporal patterns, movement, and changes over time.

    {
        query           : str,
        visual_analysis : str,
        report          : str,           # 4-section structured report
        sources         : [{title, url}],
        research_papers : [{title, url}],
        evidence: {
            evidence_level    : "strong" | "moderate" | "limited",
            evidence_reason   : str,
            source_count      : int,
            has_peer_reviewed : bool,
            confidence_note   : str,
        },
        disclaimer      : str,
        consultation_id : str,           # history row id (for PDF + feedback)
    }
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="A research query is required.")

    _cleanup_old_audio()
    owner = _resolve_user_id(request, user_id)
    has_image = bool(image and image.filename)
    has_video = bool(video and video.filename)
    started = time.perf_counter()

    request_dir = Path(tempfile.mkdtemp(prefix="skin-research-"))
    try:
        image_path = await _save_image_upload(image, request_dir)
        video_path = await _save_video_upload(video, request_dir)

        try:
            result = await run_in_threadpool(
                run_skin_research_pipeline,
                query.strip(),
                image_path,
                video_path,
            )
            # Lesion-overlay annotation for research images (same as consult).
            if image_path is not None:
                try:
                    with open(image_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
                    anno_res = await run_in_threadpool(
                        annotate_image, img_bytes, mime, image_path.name
                    )
                    if anno_res.get("annotated_image_url"):
                        result["annotated_image_url"] = anno_res["annotated_image_url"]
                        result["annotated_regions"] = anno_res.get("regions", [])
                        if anno_res.get("file_path"):
                            result["_annotated_file_path"] = anno_res["file_path"]
                except Exception as anno_err:
                    logger.warning("Research image annotation failed: %s", anno_err)
            # Persist to consultation history (text + archived still photo).
            latency_ms = int((time.perf_counter() - started) * 1000)
            await _save_consultation_record(
                result,
                save_kwargs={
                    "user_id": owner,
                    "kind": "research",
                    "input_text": query.strip(),
                    "transcript": query.strip(),
                    "report": result.get("report", ""),
                    "visual_analysis": result.get("visual_analysis", ""),
                    "evidence": result.get("evidence") or {},
                    "sources": [
                        *(result.get("sources") or []),
                        *(result.get("research_papers") or []),
                    ],
                    "media_kind": _media_kind(has_image=has_image, has_video=has_video),
                    "latency_ms": latency_ms,
                    "model_meta": _model_meta(),
                },
                image_path=image_path,
                log_label="Research",
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VisionAnalysisError as exc:
            if getattr(exc, "code", "") in {"invalid_media", "invalid_media_type"}:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.exception("Research vision step failed")
            raise HTTPException(
                status_code=502,
                detail="Visual analysis could not be completed. Please try again.",
            ) from exc
        except ConfigurationError as exc:
            logger.exception("Research misconfigured: %s", exc.code)
            raise HTTPException(
                status_code=500,
                detail="The research service is misconfigured. Please try again later.",
            ) from exc
        except Exception as exc:
            from common.errors import SearchError

            logger.exception("Research pipeline failed")
            if isinstance(exc, SearchError):
                detail = "Research search failed. Please check TAVILY_API_KEY and try again."
            elif isinstance(exc, GuidanceError):
                detail = "Visual analysis could not be completed. Please try again."
            else:
                detail = "The research pipeline could not complete this request."
            raise HTTPException(
                status_code=502,
                detail=detail,
            ) from exc
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


def _cleanup_old_audio(max_age_seconds: int = AUDIO_RETENTION_SECONDS) -> None:
    """Delete generated response mp3s older than max_age_seconds."""
    cutoff = time.time() - max_age_seconds
    for audio_file in AUDIO_DIR.glob("doctor_response_*.mp3"):
        try:
            if audio_file.stat().st_mtime < cutoff:
                audio_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not clean up %s: %s", audio_file, exc)


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    allowed_suffixes: set[str],
    max_bytes: int,
    field_name: str,
) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        content_type = (upload.content_type or "").split(";", 1)[0].lower()
        suffix = CONTENT_TYPE_SUFFIXES.get(content_type, "")
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=415, detail=f"Unsupported {field_name} media format.")

    destination = destination.with_suffix(suffix)

    total_bytes = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded media is too large.")
                output.write(chunk)
    finally:
        await upload.close()

    return destination


def _validate_image(image_path: Path) -> None:
    """Validate image file format and dimensions."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if max(image.size) > MAX_IMAGE_DIMENSION:
                raise ValueError("Image dimensions are too large.")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc


def _validate_video(video_path: Path) -> None:
    """Validate video file size and format."""
    # Basic validation - check file size
    if video_path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("Video file is too large.")

    # Check file extension
    if video_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Unsupported video format.")


async def _save_image_upload(
    image: UploadFile | None, request_dir: Path
) -> Path | None:
    """Save + validate an optional image upload. Returns path or None."""
    if not image or not image.filename:
        return None
    image_path = await _save_upload(
        image,
        request_dir / f"image{Path(image.filename).suffix.lower()}",
        ALLOWED_IMAGE_SUFFIXES,
        MAX_IMAGE_BYTES,
        "image",
    )
    try:
        if image_path:
            _validate_image(image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image_path


async def _save_video_upload(
    video: UploadFile | None, request_dir: Path
) -> Path | None:
    """Save + validate an optional video upload. Returns path or None."""
    if not video or not video.filename:
        return None
    video_path = await _save_upload(
        video,
        request_dir / f"video{Path(video.filename).suffix.lower()}",
        ALLOWED_VIDEO_SUFFIXES,
        MAX_VIDEO_BYTES,
        "video",
    )
    try:
        if video_path:
            _validate_video(video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return video_path


async def _save_audio_upload(
    audio: UploadFile | None, request_dir: Path
) -> Path | None:
    """Save an optional audio upload. Returns path or None."""
    if not audio or not audio.filename:
        return None
    return await _save_upload(
        audio,
        request_dir / f"audio{Path(audio.filename or '').suffix.lower()}",
        ALLOWED_AUDIO_SUFFIXES,
        MAX_AUDIO_BYTES,
        "audio",
    )


async def _archive_image_best_effort(
    consultation_id: str, image_path: Path | None, *, log_label: str
) -> None:
    """Archive the uploaded still image for PDF export; never raises."""
    if image_path is None:
        return
    try:
        await run_in_threadpool(
            history_store.archive_consultation_image,
            consultation_id,
            image_path,
        )
    except Exception:
        logger.warning("%s photo archive failed", log_label, exc_info=True)


async def _persist_annotated_best_effort(
    consultation_id: str, annotated_path: str | Path | None, *, log_label: str
) -> str:
    """Persist the overlay image path on the history row; never raises."""
    if not annotated_path:
        return ""
    try:
        stored = await run_in_threadpool(
            history_store.save_annotated_image_path,
            consultation_id,
            annotated_path,
        )
        return stored
    except Exception:
        logger.warning("%s annotated persist failed", log_label, exc_info=True)
        return ""


async def _save_consultation_record(
    result: dict,
    *,
    save_kwargs: dict,
    image_path: Path | None,
    log_label: str,
) -> dict:
    """Persist a consultation/research row + archive photo; mutates ``result``.

    Shared by ``/api/analyze`` and ``/api/research``. Never raises —
    persistence failures only produce a warning (request still succeeds).
    Also persists the lesion-overlay image (``annotated_image_path``) so the
    UI history + PDF export can show it.
    """
    try:
        stored = await run_in_threadpool(
            history_store.save_consultation, **save_kwargs
        )
        result["consultation_id"] = stored["id"]
        await _archive_image_best_effort(
            stored["id"], image_path, log_label=log_label
        )
        annotated_src = result.get("annotated_file_path") or result.get("_annotated_file_path")
        if annotated_src:
            persisted = await _persist_annotated_best_effort(
                stored["id"], annotated_src, log_label=log_label
            )
            if persisted:
                # Keep the public URL stable for the immediate response.
                result["annotated_image_path"] = persisted
                if not result.get("annotated_image_url"):
                    result["annotated_image_url"] = f"/audio/{Path(persisted).name}"
        # Internal-only helper key must never leak to API clients.
        result.pop("_annotated_file_path", None)
        result.pop("annotated_file_path", None)
    except Exception:
        logger.warning("%s history persist failed", log_label, exc_info=True)
    return result


def _run_analysis(
    audio_path: Path | None,
    image_path: Path | None,
    video_path: Path | None,
    text_query: str = "",
    lang_code: str = "en",
) -> dict[str, Any]:
    transcript_out = ""
    if audio_path is not None:
        transcript_out = transcribe_patient_voice(audio_path)
        if not transcript_out or not transcript_out.strip():
            raise TranscriptionError(
                "No speech could be transcribed from the audio.", code="no_speech"
            )
    elif text_query and text_query.strip():
        transcript_out = text_query.strip()
    else:
        transcript_out = ""

    # Brain needs non-empty input; transcript stays "" for pure image-only calls.
    brain_input = transcript_out.strip() or (
        "No written description provided. Please analyze the visual findings."
    )
    if len(brain_input) > 12_000:
        raise ValueError("The transcription is too long to analyze safely.")

    # Multimodal image annotation (lesion callouts + risk highlights).
    annotated_image_url = None
    annotated_file_path = None
    annotated_regions = []
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
            anno_res = annotate_image(img_bytes, mime_type=mime, output_filename=image_path.name)
            annotated_image_url = anno_res.get("annotated_image_url")
            annotated_file_path = anno_res.get("file_path")
            annotated_regions = anno_res.get("regions", [])
        except Exception as anno_err:
            logger.warning("Multimodal image annotation failed: %s", anno_err)

    # RAG vector store evidence retrieval
    rag_sources = []
    try:
        rag_sources = get_dermatology_rag().query(f"{brain_input}", top_k=3)
    except Exception as rag_err:
        logger.warning("RAG evidence retrieval failed: %s", rag_err)

    doctor_text = brain_of_the_doctor(
        patient_text=brain_input,
        image_filepath=image_path,
        video_filepath=video_path,
        lang_code=lang_code,
    )
    if not doctor_text or not doctor_text.strip():
        raise GuidanceError(
            "The analysis provider returned an empty response.",
            code="empty_response",
        )

    audio_output = AUDIO_DIR / f"doctor_response_{uuid4().hex}.mp3"
    try:
        convert_text_to_doctor_audio(doctor_text, output_filepath=audio_output)
        if not audio_output.is_file() or audio_output.stat().st_size == 0:
            raise SpeechSynthesisError(
                "Text-to-speech returned an empty audio file.",
                code="tts_empty_response",
            )
    except (SpeechSynthesisError, ConfigurationError):
        audio_output.unlink(missing_ok=True)
        raise
    except Exception as exc:
        audio_output.unlink(missing_ok=True)
        raise SpeechSynthesisError(
            "Voice generation failed.", code="tts_provider_error"
        ) from exc

    res = {
        "transcript": transcript_out,
        "guidance": doctor_text,
        "audio_url": f"/audio/{audio_output.name}",
        "language": lang_code,
        "rag_sources": rag_sources,
    }
    if annotated_image_url:
        res["annotated_image_url"] = annotated_image_url
        res["annotated_regions"] = annotated_regions
        # Internal absolute path — consumed by _save_consultation_record to
        # persist annotated_image_path, then stripped before responding.
        if annotated_file_path:
            res["_annotated_file_path"] = annotated_file_path
    return res


@app.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze(
    request: Request,
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    text: str = Form(""),
    user_id: str = Form(""),
) -> dict[str, Any]:
    has_audio = bool(audio and audio.filename)
    has_text = bool(text and text.strip())
    has_image = bool(image and image.filename)
    has_video = bool(video and video.filename)
    if not (has_audio or has_text or has_image or has_video):
        raise HTTPException(
            status_code=400,
            detail="Provide a voice description, text, image, or video.",
        )

    _cleanup_old_audio()
    owner = _resolve_user_id(request, user_id)
    lang_code = parse_accept_language(request.headers.get("accept-language"))
    started = time.perf_counter()

    request_dir = Path(tempfile.mkdtemp(prefix="skin-specialist-"))
    try:
        audio_path = await _save_audio_upload(audio, request_dir)
        image_path = await _save_image_upload(image, request_dir)
        video_path = await _save_video_upload(video, request_dir)

        try:
            result = await run_in_threadpool(
                _run_analysis, audio_path, image_path, video_path, text or "", lang_code
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            await _save_consultation_record(
                result,
                save_kwargs=dict(
                    user_id=owner,
                    kind="consult",
                    input_text=(text or "").strip(),
                    transcript=result.get("transcript", ""),
                    guidance=result.get("guidance", ""),
                    audio_url=result.get("audio_url", ""),
                    media_kind=_media_kind(
                        has_audio=has_audio, has_image=has_image, has_video=has_video
                    ),
                    latency_ms=latency_ms,
                    model_meta=_model_meta(),
                ),
                image_path=image_path,
                log_label="Consult",
            )
            return result
        except (TranscriptionError, GuidanceError, SpeechSynthesisError, ConfigurationError, VisionAnalysisError):
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Analysis pipeline failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred during analysis.",
            ) from exc
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


@app.post("/api/analyze-stream")
@limiter.limit("5/minute")
async def analyze_stream(
    request: Request,
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    text: str = Form(""),
    user_id: str = Form(""),
):
    """
    Server-Sent Events (SSE) streaming endpoint for progressive real-time analysis.
    Yields events: language_detected -> stt_transcript -> rag_evidence -> image_annotation -> guidance -> audio_url -> complete.
    """
    has_audio = bool(audio and audio.filename)
    has_text = bool(text and text.strip())
    has_image = bool(image and image.filename)
    has_video = bool(video and video.filename)
    if not (has_audio or has_text or has_image or has_video):
        raise HTTPException(
            status_code=400,
            detail="Provide a voice description, text, image, or video.",
        )

    lang_code = parse_accept_language(request.headers.get("accept-language"))
    request_dir = Path(tempfile.mkdtemp(prefix="skin-specialist-stream-"))

    audio_path = await _save_audio_upload(audio, request_dir)
    image_path = await _save_image_upload(image, request_dir)
    video_path = await _save_video_upload(video, request_dir)

    async def event_generator():
        try:
            # 1. Language event
            yield f"event: language_detected\ndata: {json.dumps({'lang': lang_code})}\n\n"

            # 2. STT Event
            transcript_out = ""
            if audio_path is not None:
                transcript_out = await run_in_threadpool(transcribe_patient_voice, audio_path)
            elif text and text.strip():
                transcript_out = text.strip()

            yield f"event: stt_transcript\ndata: {json.dumps({'transcript': transcript_out})}\n\n"

            # 3. Image Annotation Event
            if image_path:
                try:
                    with open(image_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    mime = "image/png" if str(image_path).endswith(".png") else "image/jpeg"
                    anno_res = annotate_image(img_bytes, mime_type=mime, output_filename=image_path.name)
                    yield f"event: image_annotation\ndata: {json.dumps(anno_res)}\n\n"
                except Exception as e:
                    logger.warning("SSE Image annotation failed: %s", e)

            # 4. RAG Event
            brain_input = transcript_out.strip() or "No written description provided. Please analyze visual findings."
            try:
                rag_sources = get_dermatology_rag().query(brain_input, top_k=3)
                yield f"event: rag_evidence\ndata: {json.dumps({'sources': rag_sources})}\n\n"
            except Exception as e:
                logger.warning("SSE RAG retrieval failed: %s", e)

            # 5. LLM Guidance Event
            doctor_text = await run_in_threadpool(
                brain_of_the_doctor, brain_input, image_path, video_path, lang_code
            )
            yield f"event: guidance\ndata: {json.dumps({'guidance': doctor_text})}\n\n"

            # 6. Audio Generation Event
            audio_output = AUDIO_DIR / f"doctor_response_{uuid4().hex}.mp3"
            await run_in_threadpool(convert_text_to_doctor_audio, doctor_text, audio_output)
            yield f"event: audio_url\ndata: {json.dumps({'audio_url': f'/audio/{audio_output.name}'})}\n\n"

            # 7. Complete Event
            yield f"event: complete\ndata: {json.dumps({'status': 'success'})}\n\n"

        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
        finally:
            shutil.rmtree(request_dir, ignore_errors=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Consultation History / PDF Export / Feedback Loop
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def list_history(
    request: Request,
    user_id: str = Query(""),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    kind: str | None = Query(None),
) -> dict:
    """Secure per-user listing of past sessions, newest first."""
    owner = _resolve_user_id(request, user_id)
    try:
        items, total = await run_in_threadpool(
            history_store.list_consultations, owner,
            limit=limit, offset=offset, kind=kind,
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    # List view: truncate long bodies to keep payloads small.
    summaries = []
    for item in items:
        body = item.get("report") or item.get("guidance") or ""
        summary = dict(item)
        summary["preview"] = body[:280]
        summary["report"] = ""
        summary["guidance"] = ""
        summaries.append(summary)
    return {"user_id": owner, "total": total, "limit": limit, "offset": offset, "items": summaries}


@app.get("/api/history/{consultation_id}")
async def get_history_item(
    consultation_id: str, request: Request, user_id: str = Query("")
) -> dict:
    owner = _resolve_user_id(request, user_id)
    try:
        record = await run_in_threadpool(
            history_store.get_consultation, consultation_id, owner
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    return record


@app.delete("/api/history/{consultation_id}")
async def delete_history_item(
    consultation_id: str, request: Request, user_id: str = Query("")
) -> dict:
    owner = _resolve_user_id(request, user_id)
    try:
        deleted = await run_in_threadpool(
            history_store.delete_consultation, consultation_id, owner
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    return {"deleted": True, "consultation_id": consultation_id}


@app.get("/api/history/{consultation_id}/export.pdf")
async def export_history_pdf(
    consultation_id: str, request: Request, user_id: str = Query("")
) -> Response:
    """Clinician-ready PDF: patient photo + synthesized sections + sources + evidence + disclaimer."""
    owner = _resolve_user_id(request, user_id)
    try:
        record = await run_in_threadpool(
            history_store.get_consultation, consultation_id, owner
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    try:
        pdf_bytes = await run_in_threadpool(build_consultation_pdf, record)
    except Exception as exc:
        logger.exception("PDF export failed")
        raise HTTPException(status_code=502, detail="PDF generation failed.") from exc
    short = consultation_id[:8]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="skin-report-{short}.pdf"'},
    )


@app.post("/api/feedback", status_code=201)
async def submit_feedback(request: Request, payload: FeedbackIn) -> dict:
    """Thumbs up/down + optional note. Auto-attaches run metadata for evals."""
    owner = _resolve_user_id(request, payload.user_id)
    # Ownership check first so users can't rate other users' sessions.
    try:
        parent = await run_in_threadpool(
            history_store.get_consultation, payload.consultation_id, owner
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    if parent is None:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    run_metadata = {
        "kind": parent.get("kind"),
        "media_kind": parent.get("media_kind"),
        "latency_ms": parent.get("latency_ms"),
        "evidence_level": (parent.get("evidence") or {}).get("evidence_level"),
        "source_count": len(parent.get("sources") or []),
        "report_len": len(parent.get("report") or parent.get("guidance") or ""),
        "model_meta": parent.get("model_meta") or {},
    }
    try:
        stored = await run_in_threadpool(
            history_store.save_feedback,
            consultation_id=payload.consultation_id,
            user_id=owner,
            rating=payload.rating,
            note=payload.note,
            run_metadata=run_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    return stored


@app.get("/api/feedback")
async def read_feedback(
    request: Request,
    consultation_id: str | None = Query(None),
    user_id: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Query feedback; scope to a consultation or a user for review."""
    owner = sanitize_user_id(user_id) if user_id or consultation_id else ""
    scope_user = owner or None
    # If a consultation is given without explicit user, enforce ownership via header.
    if consultation_id and not user_id:
        header_user = _resolve_user_id(request, "")
        parent = await run_in_threadpool(
            history_store.get_consultation, consultation_id, header_user
        )
        if parent is None and header_user != "anonymous":
            # Fall back to unscoped read only for anonymous legacy rows.
            pass
    try:
        items, total = await run_in_threadpool(
            history_store.list_feedback,
            consultation_id=consultation_id, user_id=scope_user,
            limit=limit, offset=offset,
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/api/evals/export")
async def export_evals(limit: int = Query(1000, ge=1, le=5000)) -> Response:
    """JSONL dump of feedback × run context for the future automated eval harness."""
    try:
        body = await run_in_threadpool(history_store.export_eval_jsonl, limit=limit)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="History store unavailable.") from exc
    return Response(
        content=body,
        media_type="application/jsonl",
        headers={"Content-Disposition": 'attachment; filename="skin-evals.jsonl"'},
    )


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_followup(request: Request, payload: ChatIn) -> dict:
    """ChatGPT-style follow-up Q&A grounded in a stored history session.

    Body: {message (required, ≤2000 chars), consultation_id? (grounds the
    reply in that session), user_id?, history? (last ≤20 turns for context)}.
    Ownership of consultation_id is enforced per user_id; cross-user ids 404.
    """
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="A message is required.")
    owner = _resolve_user_id(request, payload.user_id)

    record: dict | None = None
    if payload.consultation_id:
        try:
            record = await run_in_threadpool(
                history_store.get_consultation, payload.consultation_id, owner
            )
        except OSError as exc:
            raise HTTPException(status_code=503, detail="History store unavailable.") from exc
        if record is None:
            raise HTTPException(status_code=404, detail="Consultation not found.")

    turns = []
    for turn in (payload.history or []):
        if turn.role in ("user", "assistant"):
            role = turn.role
        else:
            role = "user"
        turns.append({"role": role, "content": turn.content})
    prompt = _build_chat_prompt(message, record, turns)
    try:
        reply = await run_in_threadpool(_run_chat_reply, prompt)
    except ConfigurationError as exc:
        logger.exception("Chat misconfigured: %s", exc.code)
        raise HTTPException(
            status_code=500,
            detail="The chat service is misconfigured. Please try again later.",
        ) from exc
    except GuidanceError as exc:
        if getattr(exc, "code", "") == "quota_exhausted":
            raise HTTPException(
                status_code=429,
                detail="The guidance provider is rate-limited. Please try again shortly.",
            ) from exc
        logger.warning("Chat guidance failed (code=%s)", exc.code)
        raise HTTPException(
            status_code=502,
            detail="A follow-up reply could not be generated. Please try again shortly.",
        ) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502,
            detail="A follow-up reply could not be generated. Please try again shortly.",
        ) from exc
    return {"reply": reply, "consultation_id": payload.consultation_id}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )