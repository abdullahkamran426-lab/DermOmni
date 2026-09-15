# `common/` — Shared Foundation

Shared, provider-agnostic helpers for the DermOmni backend.
All FastAPI routes (`main.py`), the consult brain (`brain_of_the_doctor_gemini.py`),
and the research pipeline (`Skin_research_*`) import from here instead of
re-implementing logic.

Design rules:

- No live API calls on import. Factories / pure functions only.
- Typed errors in `errors.py` — never match on exception message strings.
- Media normalization lives only in `media.py`.
- Gemini key rotation + File API lives only in `gemini.py`.
- SQLite access lives only in `history_store.py` (stdlib `sqlite3`, WAL mode).
- Raw audio/video bytes are never stored — only one normalized still photo per session.

## Folder Structure

```text
common/
├── README.md          # this file
├── __init__.py        # package marker (no logic)
├── cache.py           # process-local TTL cache + Tavily cache-key builder
├── contracts.py       # Pydantic response contracts (Source, Evidence, ResearchState)
├── env.py             # int/float env-var parsers (single source of truth)
├── errors.py          # typed exception hierarchy (maps to HTTP codes)
├── gemini.py          # Gemini key rotation, transient retry, File API upload
├── history_store.py   # SQLite consultations + feedback + eval JSONL export
├── media.py           # image/video detection, PIL normalize, data-URL encode
└── pdf_report.py      # ReportLab clinician-ready PDF export (pure function)
```

## File Guide

### `__init__.py`

Package marker only. No code, no exports. Keeps `common.*` imports stable.

### `cache.py` (~67 lines)

Small thread-safe, process-local TTL cache. Used to avoid repeat Tavily billing.

- `CACHE_VERSION = "v2"` — bump to invalidate all cached searches.
- `CacheEntry(value, expires_at)` — frozen dataclass.
- `TTLCache[T](ttl_seconds=300.0, max_entries=256)` — `get(key)`, `set(key, value)`. Evicts oldest-expiry entry when full. Not a cross-user store.
- `build_search_cache_key(*, operation, query, model, search_depth, max_results, domains)` — normalized SHA-256 key (lowercased query, sorted domains).

Used by: `Skin_research_tools.py:_retry_tavily_search` (`_SEARCH_CACHE`, 300s/256 entries).

### `contracts.py` (~48 lines)

Pydantic validation for the public research-pipeline contract.

- `Source(title, url: HttpUrl)` — one cited source. `extra="ignore"`.
- `Evidence(evidence_level: strong|moderate|limited, evidence_reason, source_count>=0, has_peer_reviewed: bool, confidence_note)` — evidence grade.
- `ResearchState(query, visual_analysis, visual_analysis_available, visual_analysis_error?, report, report_generated_by: pending|provider|local_fallback, report_error?, sources[], research_papers[], evidence, disclaimer)` — full pipeline output. `normalize_query` strips/validates `query`.

Used by: `Skin_research_pipeline.py` (validates before returning to `/api/research`).

### `env.py` (~26 lines)

Single source of truth for env parsing. Replaces the former duplicates in `voice_of_the_doctor.py` and `voice_of_the_patient.py`.

- `read_int_env(name, default, minimum=1) -> int`
- `read_float_env(name, default, minimum=1.0) -> float`

Clamps to `minimum`, falls back to `default` on missing/malformed values. Used for `DEEPGRAM_TIMEOUT_SECONDS`, `DEEPGRAM_MAX_ATTEMPTS`, `DEEPGRAM_STT_MAX_ATTEMPTS`.

### `errors.py` (~57 lines)

Typed hierarchy so `main.py` can map to HTTP codes without substring matching:

```text
RuntimeError
├── SkinResearchError
│   ├── VisionAnalysisError      (default code=vision_provider_error)
│   └── SearchError              (default code=search_provider_error)
└── ProviderError                (base: message + code)
    ├── ConfigurationError       → HTTP 500 (missing_api_key)
    ├── TranscriptionError       → HTTP 502 (no_speech / stt_*)
    ├── SpeechSynthesisError     → HTTP 502 (tts_*)
    └── GuidanceError            → HTTP 502 / 429 when code=quota_exhausted
```

Every error carries `.code`. Raised in `brain_*`, `voice_*`, `Skin_research_tools`, `huggingface_vision`, `gemini.py`; caught in `main.py` + `Skin_research_pipeline.py`.

### `gemini.py` (~292 lines)

Multi-key rotation + File API. Single source of truth for `GEMINI_API_KEY` (primary) + `GEMINI_API_KEYS` (comma/newline/semicolon-separated backups).

- Env: `GEMINI_API_KEY`, `GEMINI_API_KEYS`, `GEMINI_FILE_API_THRESHOLD_BYTES` (default 10MB), `GEMINI_FILE_POLL_TIMEOUT` (60s), `GEMINI_FILE_POLL_INTERVAL` (2s).
- `get_gemini_api_keys() -> list[str]` — primary first, deduped. Raises `ConfigurationError` when empty.
- `is_quota_error(exc)` — 429 / `resource_exhausted` / `quota` / `rate limit`.
- `is_invalid_key_error(exc)` — 400/401/403 + key markers (`api key not valid`, `unauthenticated`, …).
- `should_rotate_key(exc)` — quota OR bad-key.
- `is_transient_error(exc)` — 500/502/503/504 / `overloaded` / `unavailable`.
- `call_with_key_rotation(operation, client_factory, *, operation_name, max_transient_retries=2)` — tries each key in turn; transient 503 retried on same key with 1s/2s backoff. `execute_with_key_rotation` is a test-compat alias.
- `needs_file_api(media_path)` — videos always `True`; images only when `> FILE_API_THRESHOLD_BYTES`.
- `upload_file_and_wait(client, media_path, *, mime_type)` — `files.upload` + poll until `ACTIVE/READY`; raises `VisionAnalysisError` on `FAILED/ERROR`.
- `delete_remote_file(client, uploaded)` — best-effort cleanup (never raises).

Used by: `brain_of_the_doctor_gemini.py`, `Skin_research_tools.py`, `Skin_research_agents.py`, `main.py:_run_chat_reply`.

### `history_store.py` (~509 lines)

Secure per-user SQLite store (stdlib only, WAL, thread-locked writes). Respects `HISTORY_DB_PATH` override (tests use tmp DBs).

Tables:

- `consultations(id, user_id, kind: consult|research, created_at, input_text, transcript, guidance, report, visual_analysis, audio_url, evidence_json, sources_json, media_kind: none|image|video|audio|mixed, media_image_path, latency_ms, model_meta_json)`
- `feedback(id, consultation_id FK, user_id, rating: up|down, note<=2000, run_metadata_json, created_at)`

Key functions:

- `get_db_path()`, `get_media_dir()` — `data/consultations.db` + `data/consultation_media/` by default.
- `init_db()` — idempotent schema + migration for `media_image_path`.
- `save_consultation(*, user_id, kind, input_text, transcript, guidance, report, visual_analysis, audio_url, evidence, sources, media_kind, latency_ms, model_meta)` — returns row dict.
- `list_consultations(user_id, limit<=100, offset, kind?) → (items, total)` — newest first.
- `get_consultation(id, user_id)` — ownership-enforced read.
- `delete_consultation(id, user_id) -> bool` — also deletes archived photo.
- `archive_consultation_image(record_id, src_path)` — EXIF-correct RGB thumbnail 1024px q82 to `consultation_media/<id>.jpg`; never raises (returns `""` on skip).
- `sanitize_user_id(raw)` — alnum + `-_:@.` max 64 chars, else `"anonymous"`.
- `save_feedback(*, consultation_id, user_id, rating: up|down|1|-1, note, run_metadata)` — normalizes rating, validates parent exists.
- `list_feedback(*, consultation_id?, user_id?, limit<=200, offset)` — review queries.
- `export_eval_jsonl(limit<=5000)` — feedback × parent context JSONL for future eval harness.

Used by: all `/api/history*`, `/api/feedback`, `/api/evals/export`, `/api/analyze`, `/api/research`, `/api/chat` in `main.py`.

### `media.py` (~130 lines)

Single entry point for vision payload prep. All Gemini/HF paths must call this.

- Constants: `VISION_MAX_DIMENSION=2048, VISION_JPEG_QUALITY=90` (Gemini inline), `FALLBACK_MAX_DIMENSION=1024, FALLBACK_JPEG_QUALITY=85` (HF base64), `VIDEO_EXTENSIONS={.mp4,.webm,.mov,.avi,.mkv,.flv,.wmv,.3gpp}`.
- `detect_media_type(path) -> "video"|"image"` — extension-based.
- `get_video_mime_type(ext)` — extension → MIME map.
- `prepare_vision_media(path) -> (bytes, mime, "image"|"video")` — images: EXIF-correct, alpha-flatten to white, thumbnail, JPEG-encode; videos: raw bytes + MIME. Raises `VisionAnalysisError(invalid_media)`.
- `encode_image_to_data_url(path, *, max_dimension, quality)` — base64 `data:image/jpeg` for HF chat-completion fallback. Rejects video with `invalid_media_type`.
- `prepare_vision_image(path) -> (bytes, mime)` — legacy image-only wrapper (unused in prod, kept for compat).
- `_prepare_image_bytes(...)` — internal PIL worker.

Used by: `brain_of_the_doctor_gemini.py`, `Skin_research_tools.py`, `huggingface_vision.py`.

### `pdf_report.py` (~311 lines)

Pure function PDF export (ReportLab Platypus, no temp files, no global state).

- `build_consultation_pdf(record: dict) -> bytes` — header brand bar → meta table (kind/media/evidence/sources) → amber disclaimer callout → archived patient photo (aspect-preserved, max 150×90mm) → patient description → visual analysis → 4-section report → numbered sources (max 20) → evidence box → footer. Returns PDF bytes for `Response(media_type="application/pdf")`.
- `parse_report_sections(report) -> [(heading, body)]` — splits `VISUAL OBSERVATIONS / POTENTIAL CONDITIONS / RECOMMENDATIONS / WHEN TO SEE A DOCTOR`; falls back to `[("CLINICAL GUIDANCE", text)]` for consult outputs.
- Styling: `SAGE #1F6F5C`, `AMBER #C98A2C`, A4, 18/18/16/20mm margins.

Used by: `GET /api/history/{id}/export.pdf` in `main.py`. Photo lookup via `media_image_path` (same resolution logic as `history_store.resolve_media_image`).

## Import Map (who uses what)

```text
main.py
 ├── common.history_store (init_db, save/get/list/delete, archive, sanitize)
 ├── common.gemini (call_with_key_rotation, get_gemini_api_keys, is_quota_error)
 ├── common.pdf_report (build_consultation_pdf)
 └── common.errors (ConfigurationError, GuidanceError, SpeechSynthesisError, ...)
brain_of_the_doctor_gemini.py
 ├── common.gemini + common.media + common.errors
Skin_research_tools.py
 ├── common.cache + common.errors + common.gemini + common.media
Skin_research_pipeline.py
 └── common.contracts (ResearchState, Evidence, Source)
voice_of_the_doctor.py / voice_of_the_patient.py
 ├── common.env + common.errors
huggingface_vision.py
 └── common.errors + common.media
```
