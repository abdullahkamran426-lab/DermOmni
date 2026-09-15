"""Secure consultation-history + feedback/eval store (SQLite, stdlib only).

Design goals (production-ready, minimal ops):
- Zero new DB dependencies: stdlib ``sqlite3`` with WAL mode.
- Per-user isolation: every row carries an opaque ``user_id`` (client UUID
  in localStorage, or ``X-User-Id`` header). All reads/deletes filter by it.
- Privacy: text outputs (transcript/guidance/report/visual_analysis + JSON
  metadata) plus ONE normalized still photo per session (for the PDF) are
  persisted. Raw audio/video bytes are NEVER stored — uploads live in
  per-request temp dirs and are deleted (see README).
- Thread-safe: FastAPI runs pipelines in threadpools; a module-level lock
  serializes writes. Reads use short-lived connections.
- Eval-ready: ``feedback`` rows carry ``run_metadata_json`` (model names,
  latency, evidence level, media kind) so a future harness can dump JSONL.

Schema
------
consultations(
    id TEXT PRIMARY KEY,            -- uuid4 hex
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,             -- 'consult' | 'research'
    created_at TEXT NOT NULL,       -- UTC ISO-8601
    input_text TEXT DEFAULT '',     -- typed query / transcript source
    transcript TEXT DEFAULT '',
    guidance TEXT DEFAULT '',       -- consult path output
    report TEXT DEFAULT '',         -- research path output
    visual_analysis TEXT DEFAULT '',
    audio_url TEXT DEFAULT '',
    evidence_json TEXT DEFAULT '{}',
    sources_json TEXT DEFAULT '[]', -- [{title, url}]
    media_kind TEXT DEFAULT 'none', -- 'none' | 'image' | 'video' | 'audio' | 'mixed'
    media_image_path TEXT DEFAULT '', -- archived normalized still image for PDF
    latency_ms INTEGER DEFAULT 0,
    model_meta_json TEXT DEFAULT '{}'
)

Media-photo note: the still image a patient uploads IS archived (normalized
JPEG thumbnail, max 1024px / q82, under ``consultation_media/<id>.jpg``) so it
can be embedded in the PDF export. Audio/video bytes are NEVER stored.
Deleting a consultation also deletes its archived photo (best-effort).
feedback(
    id TEXT PRIMARY KEY,
    consultation_id TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    rating TEXT NOT NULL,           -- 'up' | 'down'
    note TEXT DEFAULT '',
    run_metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_LOCK = threading.Lock()

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS consultations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('consult','research')),
    created_at TEXT NOT NULL,
    input_text TEXT DEFAULT '',
    transcript TEXT DEFAULT '',
    guidance TEXT DEFAULT '',
    report TEXT DEFAULT '',
    visual_analysis TEXT DEFAULT '',
    audio_url TEXT DEFAULT '',
    evidence_json TEXT DEFAULT '{}',
    sources_json TEXT DEFAULT '[]',
    media_kind TEXT DEFAULT 'none',
    media_image_path TEXT DEFAULT '',
    annotated_image_path TEXT DEFAULT '',
    latency_ms INTEGER DEFAULT 0,
    model_meta_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_consult_user_created
    ON consultations(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    consultation_id TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up','down')),
    note TEXT DEFAULT '',
    run_metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_consult
    ON feedback(consultation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_user
    ON feedback(user_id, created_at DESC);
"""

_ANONYMOUS = "anonymous"


def get_db_path() -> Path:
    """Resolve DB path; ``HISTORY_DB_PATH`` overrides the default."""
    override = os.getenv("HISTORY_DB_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "consultations.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path | None = None) -> Path:
    """Create tables/indexes idempotently. Returns the resolved path."""
    path = db_path or get_db_path()
    with _LOCK, _connect(path) as conn:
        conn.executescript(_SCHEMA)
        # Lightweight migrations for DBs created before newer columns existed.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(consultations)").fetchall()}
        if "media_image_path" not in cols:
            conn.execute("ALTER TABLE consultations ADD COLUMN media_image_path TEXT DEFAULT ''")
        if "annotated_image_path" not in cols:
            conn.execute("ALTER TABLE consultations ADD COLUMN annotated_image_path TEXT DEFAULT ''")
        conn.commit()
    get_media_dir(path).mkdir(parents=True, exist_ok=True)
    return path


def get_media_dir(db_path: Path | None = None) -> Path:
    """Directory holding archived normalized consultation photos.

    Derived from the DB location so ``HISTORY_DB_PATH`` overrides (incl. tmp
    test DBs) stay isolated. Absolute paths are stored in the DB.
    """
    base = db_path or get_db_path()
    return base.parent / "consultation_media"


def archive_consultation_image(
    record_id: str, src_path: str | Path | None, *, db_path: Path | None = None
) -> str:
    """Archive a normalized copy of the uploaded still image for PDF export.

    Normalizes via Pillow (EXIF-correct, RGB, thumbnail 1024px, JPEG q82) to
    ``consultation_media/<record_id>.jpg`` and UPDATEs the row. Returns the
    stored absolute path, or ``""`` when there is nothing usable to archive
    (no image, video-only, unreadable file). Never raises — PDF/text paths
    must not break because a photo could not be archived.
    """
    if not record_id or src_path is None:
        return ""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return ""
    src = Path(src_path)
    if not src.is_file() or src.stat().st_size <= 0:
        return ""
    try:
        dest_dir = get_media_dir(db_path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{record_id}.jpg"
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, "white")
                bg.paste(rgba, mask=rgba.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            im.save(dest, format="JPEG", quality=82, optimize=True)
        if not dest.is_file() or dest.stat().st_size <= 0:
            return ""
        stored = str(dest.resolve())
        with _LOCK, _connect(db_path) as conn:
            conn.execute(
                "UPDATE consultations SET media_image_path = ? WHERE id = ?",
                (stored, record_id),
            )
            conn.commit()
        return stored
    except Exception:
        return ""


def save_annotated_image_path(
    record_id: str, annotated_src: str | Path | None, *, db_path: Path | None = None
) -> str:
    """Persist the annotator's overlay file path on the consultation row.

    The overlay PNG already lives under the public ``generated_audio/`` dir
    (served at ``/audio/``), so no copy is needed — just UPDATE the row.
    Returns the stored absolute path or ``""``. Never raises.
    """
    if not record_id or annotated_src is None:
        return ""
    try:
        src = Path(annotated_src)
        if not src.is_file() or src.stat().st_size <= 0:
            return ""
        stored = str(src.resolve())
        with _LOCK, _connect(db_path) as conn:
            try:
                conn.execute(
                    "UPDATE consultations SET annotated_image_path = ? WHERE id = ?",
                    (stored, record_id),
                )
            except sqlite3.OperationalError:
                # Pre-migration DB without the column — re-init then retry once.
                conn.executescript(_SCHEMA)
                conn.execute(
                    "UPDATE consultations SET annotated_image_path = ? WHERE id = ?",
                    (stored, record_id),
                )
            conn.commit()
        return stored
    except Exception:
        return ""


def annotated_url_for_path(stored_path: str | None) -> str:
    """Derive the public ``/audio/<file>`` URL for a stored annotated path."""
    if not stored_path:
        return ""
    try:
        name = Path(stored_path).name
        if not name:
            return ""
        return f"/audio/{name}"
    except Exception:
        return ""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_user_id(raw: str | None) -> str:
    """Normalize an opaque client user id; fall back to 'anonymous'."""
    if not raw:
        return _ANONYMOUS
    allowed_characters = ("-", "_", ":", "@", ".")
    cleaned_characters = []
    for character in raw.strip():
        if character.isalnum() or character in allowed_characters:
            cleaned_characters.append(character)
    cleaned = "".join(cleaned_characters)[:64]
    if not cleaned:
        return _ANONYMOUS
    return cleaned


def _row_to_consultation(row: sqlite3.Row) -> dict:
    # sqlite3.Row -> dict keeps every column incl. media_image_path /
    # annotated_image_path (missing on pre-migration rows -> default "").
    data = dict(row)
    data.setdefault("media_image_path", "")
    data.setdefault("annotated_image_path", "")
    # Convenience public URL for UI + PDF clients (derived, not a column).
    try:
        data["annotated_image_url"] = annotated_url_for_path(data.get("annotated_image_path") or "")
    except Exception:
        data["annotated_image_url"] = ""
    for key in ("evidence_json", "sources_json", "model_meta_json"):
        if key == "sources_json":
            output_key = "sources"
            empty_value: dict | list = []
            fallback_raw = "[]"
        else:
            output_key = key.replace("_json", "")
            empty_value = {}
            fallback_raw = "{}"
        raw = data.get(key) or fallback_raw
        try:
            data[output_key] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data[output_key] = empty_value
        data.pop(key, None)
    # Back-compat aliases used by API/PDF layers.
    evidence = data.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    data["evidence"] = evidence
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    data["sources"] = sources
    model_meta = data.get("model_meta") or {}
    # model_meta column is named model_meta_json -> key becomes model_meta
    if not isinstance(model_meta, dict):
        model_meta = {}
    data["model_meta"] = model_meta
    return data


def save_consultation(
    *,
    user_id: str,
    kind: str,
    input_text: str = "",
    transcript: str = "",
    guidance: str = "",
    report: str = "",
    visual_analysis: str = "",
    audio_url: str = "",
    evidence: dict | None = None,
    sources: list[dict] | None = None,
    media_kind: str = "none",
    media_image_path: str = "",
    annotated_image_path: str = "",
    latency_ms: int = 0,
    model_meta: dict | None = None,
    db_path: Path | None = None,
) -> dict:
    """Persist one consultation/research run; returns the stored row as dict."""
    if kind not in ("consult", "research"):
        raise ValueError("kind must be 'consult' or 'research'")
    record_id = uuid4().hex
    created_at = utcnow_iso()
    payload = (
        record_id,
        sanitize_user_id(user_id),
        kind,
        created_at,
        input_text or "",
        transcript or "",
        guidance or "",
        report or "",
        visual_analysis or "",
        audio_url or "",
        json.dumps(evidence or {}, ensure_ascii=False),
        json.dumps(sources or [], ensure_ascii=False),
        media_kind or "none",
        media_image_path or "",
        annotated_image_path or "",
        max(0, int(latency_ms or 0)),
        json.dumps(model_meta or {}, ensure_ascii=False),
    )
    with _LOCK, _connect(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO consultations (id, user_id, kind, created_at, input_text,"
                " transcript, guidance, report, visual_analysis, audio_url,"
                " evidence_json, sources_json, media_kind, media_image_path,"
                " annotated_image_path, latency_ms, model_meta_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        except sqlite3.OperationalError:
            # Pre-migration DB without annotated_image_path — migrate then retry.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(consultations)").fetchall()}
            if "annotated_image_path" not in cols:
                conn.execute("ALTER TABLE consultations ADD COLUMN annotated_image_path TEXT DEFAULT ''")
            if "media_image_path" not in cols:
                conn.execute("ALTER TABLE consultations ADD COLUMN media_image_path TEXT DEFAULT ''")
            conn.execute(
                "INSERT INTO consultations (id, user_id, kind, created_at, input_text,"
                " transcript, guidance, report, visual_analysis, audio_url,"
                " evidence_json, sources_json, media_kind, media_image_path,"
                " annotated_image_path, latency_ms, model_meta_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        conn.commit()
        row = conn.execute("SELECT * FROM consultations WHERE id = ?", (record_id,)).fetchone()
    return _row_to_consultation(row)


def list_consultations(
    user_id: str,
    *,
    limit: int = 20,
    offset: int = 0,
    kind: str | None = None,
    db_path: Path | None = None,
) -> tuple[list[dict], int]:
    """Return ``(items, total)`` for a user, newest first."""
    limit = min(max(1, int(limit or 20)), 100)
    offset = max(0, int(offset or 0))
    user_id = sanitize_user_id(user_id)
    where = "WHERE user_id = ?"
    params: list = [user_id]
    if kind in ("consult", "research"):
        where += " AND kind = ?"
        params.append(kind)
    with _connect(db_path) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM consultations {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM consultations {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [_row_to_consultation(r) for r in rows], int(total)


def get_consultation(
    consultation_id: str, user_id: str, *, db_path: Path | None = None
) -> dict | None:
    """Fetch one row scoped to ``user_id`` (ownership enforced)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM consultations WHERE id = ? AND user_id = ?",
            (consultation_id, sanitize_user_id(user_id)),
        ).fetchone()
    return _row_to_consultation(row) if row else None


def delete_consultation(consultation_id: str, user_id: str, *, db_path: Path | None = None) -> bool:
    with _LOCK, _connect(db_path) as conn:
        try:
            row = conn.execute(
                "SELECT media_image_path, annotated_image_path FROM consultations WHERE id = ? AND user_id = ?",
                (consultation_id, sanitize_user_id(user_id)),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                "SELECT media_image_path FROM consultations WHERE id = ? AND user_id = ?",
                (consultation_id, sanitize_user_id(user_id)),
            ).fetchone()
        cur = conn.execute(
            "DELETE FROM consultations WHERE id = ? AND user_id = ?",
            (consultation_id, sanitize_user_id(user_id)),
        )
        conn.commit()
        deleted = cur.rowcount > 0
    if deleted and row is not None:
        try:
            mapping = dict(row)
            photo = (mapping.get("media_image_path") or "").strip()
            if photo:
                Path(photo).unlink(missing_ok=True)
            annotated = (mapping.get("annotated_image_path") or "").strip()
            if annotated:
                Path(annotated).unlink(missing_ok=True)
        except (OSError, TypeError):
            pass
    return deleted


# ---------------------------------------------------------------------------
# Feedback + eval harness
# ---------------------------------------------------------------------------

def save_feedback(
    *,
    consultation_id: str,
    user_id: str,
    rating: str,
    note: str = "",
    run_metadata: dict | None = None,
    db_path: Path | None = None,
) -> dict:
    """Store a thumbs up/down + optional note linked to a consultation.

    ``run_metadata`` should carry eval-relevant fields such as
    ``{evidence_level, latency_ms, media_kind, model, report_len}``; the API
    layer auto-fills it from the parent consultation when omitted.
    """
    normalized = str(rating or "").strip().lower()
    if normalized in ("1", "+1", "thumbs_up", "up", "positive"):
        normalized = "up"
    elif normalized in ("-1", "thumbs_down", "down", "negative"):
        normalized = "down"
    if normalized not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    note = (note or "").strip()[:2000]
    feedback_id = uuid4().hex
    created_at = utcnow_iso()
    with _LOCK, _connect(db_path) as conn:
        parent = conn.execute(
            "SELECT id FROM consultations WHERE id = ?", (consultation_id,)
        ).fetchone()
        if parent is None:
            raise KeyError(f"unknown consultation_id: {consultation_id}")
        conn.execute(
            "INSERT INTO feedback (id, consultation_id, user_id, rating, note,"
            " run_metadata_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                feedback_id,
                consultation_id,
                sanitize_user_id(user_id),
                normalized,
                note,
                json.dumps(run_metadata or {}, ensure_ascii=False),
                created_at,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
    out = dict(row)
    try:
        out["run_metadata"] = json.loads(out.pop("run_metadata_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        out["run_metadata"] = {}
    return out


def list_feedback(
    *,
    consultation_id: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Path | None = None,
) -> tuple[list[dict], int]:
    limit = min(max(1, int(limit or 50)), 200)
    offset = max(0, int(offset or 0))
    clauses, params = [], []
    if consultation_id:
        clauses.append("consultation_id = ?")
        params.append(consultation_id)
    if user_id:
        clauses.append("feedback.user_id = ?")
        params.append(sanitize_user_id(user_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM feedback {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM feedback {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        try:
            item["run_metadata"] = json.loads(item.pop("run_metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            item["run_metadata"] = {}
        items.append(item)
    return items, int(total)


def export_eval_jsonl(*, limit: int = 1000, db_path: Path | None = None) -> str:
    """Dump feedback rows joined with parent run context as JSONL.

    Each line: ``{feedback_id, rating, note, created_at, consultation: {...},
    run_metadata: {...}}`` — ready for a future automated eval harness.
    """
    limit = min(max(1, int(limit or 1000)), 5000)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT f.*, c.kind, c.input_text, c.transcript, c.guidance, c.report,"
            " c.visual_analysis, c.evidence_json, c.sources_json, c.media_kind,"
            " c.latency_ms, c.model_meta_json, c.created_at AS run_created_at"
            " FROM feedback f JOIN consultations c ON c.id = f.consultation_id"
            " ORDER BY f.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    lines = []
    for row in rows:
        mapping = dict(row)
        try:
            run_meta = json.loads(mapping.get("run_metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            run_meta = {}
        for key in ("evidence_json", "sources_json", "model_meta_json"):
            try:
                mapping[key] = json.loads(mapping.get(key) or ("[]" if key == "sources_json" else "{}"))
            except (json.JSONDecodeError, TypeError):
                mapping[key] = [] if key == "sources_json" else {}
        lines.append(
            json.dumps(
                {
                    "feedback_id": mapping["id"],
                    "consultation_id": mapping["consultation_id"],
                    "user_id": mapping["user_id"],
                    "rating": mapping["rating"],
                    "note": mapping.get("note", ""),
                    "created_at": mapping.get("created_at"),
                    "run_metadata": run_meta,
                    "consultation": {
                        "kind": mapping.get("kind"),
                        "input_text": mapping.get("input_text", ""),
                        "transcript": mapping.get("transcript", ""),
                        "guidance": mapping.get("guidance", ""),
                        "report": mapping.get("report", ""),
                        "visual_analysis": mapping.get("visual_analysis", ""),
                        "evidence": mapping.get("evidence_json", {}),
                        "sources": mapping.get("sources_json", []),
                        "media_kind": mapping.get("media_kind"),
                        "latency_ms": mapping.get("latency_ms"),
                        "model_meta": mapping.get("model_meta_json", {}),
                        "run_created_at": mapping.get("run_created_at"),
                    },
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)
