# `tests/` — Offline QA Suite

100% offline pytest suite. No live Gemini / Deepgram / Tavily calls.
Dummy keys are set in `conftest.py` before app import, and any direct
construction of a live client fails loudly via mocks.

Run: `.venv\Scripts\python.exe -m pytest tests/ -q` (48 tests, ~2s).
Config: `pyproject.toml` → `testpaths=["tests"]`, `asyncio_mode="auto"`.

## Folder Structure

```text
tests/
├── README.md                    # this file
├── __init__.py                  # empty package marker (0 bytes)
├── conftest.py                  # dummy creds, live-client blocks, shared fixtures
├── test_endpoints.py            # POST /api/analyze + GET /health (7 tests)
├── test_chat.py                 # POST /api/chat follow-up Q&A (6 tests)
├── test_history_feedback_pdf.py # history store, feedback, PDF, photo archive (7 tests)
├── test_media.py                # media prep + inline-vs-File-API routing (9 tests)
├── test_gemini_client.py        # key rotation + 503 retry resilience (6 tests)
├── test_localization.py         # Accept-Language parsing + directives (7 tests)
├── test_dermatology_rag.py      # TF-IDF fallback engine + RAG query (2 tests)
├── test_image_annotator.py      # lesion regions + annotation canvas (2 tests)
└── test_sse.py                  # /api/analyze-stream SSE events (2 tests)
```

## File Guide

### `__init__.py`

Empty marker. Makes `tests` a package; no fixtures or logic here.

### `conftest.py` (~113 lines)

Suite guarantees + shared fixtures (imported automatically by pytest):

- Dummy creds set before `import main`: `GEMINI_API_KEY`, `GEMINI_API_KEYS`, `DEEPGRAM_API_KEY`, `TAVILY_API_KEY`, `HF_API_KEY`.
- `_block_live_clients` (autouse): patches `google.genai.Client`, `voice_of_the_patient.DeepgramClient`, `voice_of_the_doctor.DeepgramClient`, `Skin_research_tools.TavilyClient` to raise `AssertionError` on direct construction.
- `_reset_rate_limiter` (autouse): resets slowapi in-memory storage before/after each test so the 5/min `/api/analyze` limit stays deterministic.
- `client`: sync `TestClient(main.app)` for endpoint tests.
- `mock_brain`: patches `main.brain_of_the_doctor` → fixed guidance string.
- `mock_tts`: patches `main.convert_text_to_doctor_audio` → writes `b"FAKE-MP3-BYTES"` to the expected path (this is what creates `generated_audio/doctor_response_*.mp3` during tests; 14B files, gitignored).
- `sample_image_bytes`: 100×100 red JPEG in memory for upload tests.

### `test_endpoints.py` (6 tests)

`POST /api/analyze` integration + health checks. Uses `mock_brain` + `mock_tts`.

- `test_health_check` / `test_health_check_async`: `GET /health` → `{"status":"ok"}` (sync + async httpx `ASGITransport`).
- `test_analyze_valid_text`: text-only → `transcript` echo, `guidance` present, `audio_url` starts with `/audio/`.
- `test_analyze_valid_image_only`: image-only → `transcript == ""`, guidance + audio present.
- `test_analyze_combined_text_and_image`: text + image path.
- `test_analyze_missing_input_returns_400`: blank text, no files → 400.
- `test_analyze_rate_limit_returns_429`: 6 rapid posts → 5×200 then 429.

### `test_chat.py` (6 tests)

`POST /api/chat` history-grounded follow-ups. `_run_chat_reply` mocked; sessions created via real `/api/analyze` with mocked brain/TTS.

- `_make_session(client, mock_brain, mock_tts, user_id)`: helper — posts to `/api/analyze`, returns `consultation_id`.
- `test_chat_grounded_in_session`: reply 200, prompt contains stored session text + new question.
- `test_chat_general_mode_without_session`: no `consultation_id` → general reply, `consultation_id is None`.
- `test_chat_rejects_empty_message`: blank → 400.
- `test_chat_unknown_session_404` / `test_chat_cross_user_session_404`: bad id or cross-user id → 404 (ownership enforced).
- `test_chat_provider_failure_502`: `GuidanceError` from reply fn → 502.

### `test_history_feedback_pdf.py` (7 tests)

History, feedback, PDF, photo archive. `isolated_db` fixture (local): tmp `HISTORY_DB_PATH`, reloads `common.history_store`, rebinds `main.history_store`.

- `test_store_roundtrip`: save → list → get → cross-user isolation → delete.
- `test_feedback_and_eval_export`: `rating "1"` normalizes to `"up"`; bad rating → `ValueError`; unknown consult → `KeyError`; `export_eval_jsonl` contains id + rating.
- `test_pdf_builder_smoke`: 4-section report → `parse_report_sections` == 4; `build_consultation_pdf` starts with `b"%PDF-"`.
- `test_analyze_persists_history`: `/api/analyze` → `/api/history` per-user counts → owner PDF 200, intruder 404.
- `test_photo_archive_and_pdf_embed`: archived JPEG grows PDF (`/XObject`/`/Image`, +500B); delete removes photo file.
- `test_analyze_with_image_archives_photo`: upload → `media_image_path` set → PDF 200.
- `test_feedback_api_flow`: `POST /api/feedback` 201 + `run_metadata.kind == "consult"`; bad rating 400; unknown consult 404; `/api/evals/export` contains consult id.

### `test_media.py` (9 tests)

Unit tests for `common/media.py` + `common/gemini.needs_file_api` + `Skin_research_tools._run_vision_analysis` routing. Pillow/tmp files only.

- Fixtures: `small_image` (100×100 JPEG), `tiny_video` (3 KiB `.mp4`, never decoded).
- `test_detect_media_type_image_vs_video`: `.jpg` → image; `.mp4`/`.WEBM` → video.
- `test_prepare_small_image_returns_jpeg_bytes` / `test_prepare_video_returns_raw_bytes_with_mime` / `test_prepare_undecodable_image_raises` (`invalid_media`).
- `test_small_image_does_not_need_file_api` / `test_video_always_needs_file_api` / `test_large_file_needs_file_api` (threshold monkeypatched to 10B).
- `test_small_input_uses_inline_byte_parts`: `generate_content` called, `files.upload` never called.
- `test_large_input_routes_through_files_upload`: `upload_file_and_wait` + `delete_remote_file` each called once.

### `test_gemini_client.py` (7 tests)

Resilience for `common.gemini.call_with_key_rotation` (alias `execute_with_key_rotation`). `google.genai.Client` mocked per-key; `FakeApiError(code/status_code)` stands in for SDK errors.

- Fixtures: `two_keys` (`KEY-1`+`KEY-2`), `single_key` (`ONLY-KEY`).
- `test_rotation_alias_matches_implementation`: alias identity check.
- `test_failover_to_key2_on_429_resource_exhausted`: key1 429 → key2 succeeds (2 client constructions).
- `test_invalid_key_error_also_fails_over`: key1 400 `API key not valid` → key2 succeeds.
- `test_503_retries_with_exponential_backoff_then_succeeds`: one 503 → retry same key, `sleep == [1.0]`, 1 client.
- `test_503_exhausts_retries_and_propagates`: persistent 503 → raises, `sleep == [1.0, 2.0]`.
- `test_non_retryable_error_propagates_without_rotation`: `ValueError` → immediate raise, 1 client.

### `test_localization.py` (7 tests)

`Accept-Language` parsing + localized prompt/disclaimer helpers in `common/localization.py`.

- `test_parse_accept_language_*`: single tag, multi-preference (`es-ES,es;q=0.9,...` → `es`), unsupported → `en` fallback, missing/empty → `en`.
- `test_get_localized_directive` / `test_get_localized_disclaimer` / `test_get_language_name`: per-language strings with `en` fallback.

### `test_dermatology_rag.py` (2 tests)

TF-IDF fallback engine + store query in `common/dermatology_rag.py` (in-memory docs, no ChromaDB needed).

- `test_fallback_vector_engine_search`: `FallbackVectorEngine` ranks the melanoma doc first with a `relevance_score`.
- `test_dermatology_vector_store_query`: `get_dermatology_rag().query(...)` returns non-empty `title`/`content` rows.

### `test_image_annotator.py` (2 tests)

Lesion-region detection fallback + canvas rendering in `common/image_annotator.py` (in-memory PNG, Gemini mocked).

- `test_detect_skin_lesion_regions_fallback`: invalid model JSON → heuristic `Region of Interest` with `label` + `box_2d`.
- `test_annotate_image_canvas_rendering`: mocked regions → annotated PNG bytes saved under `generated_audio/`.

### `test_sse.py` (2 tests)

`POST /api/analyze-stream` SSE contract (brain + TTS mocked).

- `test_analyze_stream_no_input`: empty post → 400.
- `test_analyze_stream_text_query`: 200 `text/event-stream` with `language_detected → stt_transcript → rag_evidence → guidance → audio_url → complete` events.
