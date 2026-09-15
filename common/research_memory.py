"""
Per-user research memory for DermOmni.

Successful provider-synthesised research reports are chunked and stored in a
persistent vector DB (ChromaDB ``research_memory`` collection). When live web /
paper search is unavailable, the research pipeline reads this memory back and
feeds it to the synthesis LLM as cached evidence — a per-user RAG fallback.

Privacy: every chunk carries ``user_id`` metadata and all reads are filtered
by it. The ``"anonymous"`` pseudo-user is intentionally excluded — anonymous
sessions are never ingested and never read back, so strangers can't see each
other's cached research.

Backends (``RESEARCH_MEMORY_BACKEND``: ``auto`` | ``chroma`` | ``jsonl``):
- ``chroma``: ``chromadb.PersistentClient`` rooted at ``RESEARCH_MEMORY_PATH``
  (default ``data/research_memory``). Survives restarts.
- ``jsonl``: zero-dependency file store (``memory.jsonl`` in the same dir) +
  the TF-IDF ``FallbackVectorEngine`` from ``dermatology_rag``.
- ``auto`` (default): try Chroma, silently drop to JSONL on any failure.

Every public function is best-effort and never raises — memory must never
fail a clinical call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "research_memory"
_JSONL_FILENAME = "memory.jsonl"
_CHUNK_CHARS = 600
_MAX_CHUNKS_PER_USER = 300
_ANONYMOUS_USER = "anonymous"


def _memory_dir() -> Path:
    return Path(os.getenv("RESEARCH_MEMORY_PATH", os.path.join("data", "research_memory")))


def _backend() -> str:
    return os.getenv("RESEARCH_MEMORY_BACKEND", "auto").strip().lower() or "auto"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_text(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Split a report into paragraph-aware chunks of roughly ``limit`` chars."""
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= limit:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            current = para
    if current:
        chunks.append(current)
    return chunks


class ResearchMemory:
    """Persistent per-user store of past research, with vector retrieval."""

    def __init__(self, directory: str | Path | None = None, backend: str | None = None):
        self.directory = Path(directory) if directory else _memory_dir()
        self.backend = (backend or _backend())
        self._lock = threading.Lock()
        self._chroma_collection = None
        self._jsonl_records: list[dict[str, Any]] = []
        self._engine = None  # lazy TF-IDF engine over the JSONL records
        self._engine_size = -1
        self._init_store()

    # -- init -----------------------------------------------------------
    def _init_store(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Research memory dir unavailable (%s); memory disabled.", exc)
            return
        if self.backend in ("auto", "chroma"):
            try:
                # pyrefly: ignore [missing-import]
                import chromadb

                client = chromadb.PersistentClient(path=str(self.directory / "chroma"))
                self._chroma_collection = client.get_or_create_collection(_COLLECTION_NAME)
                logger.info("Research memory: persistent ChromaDB ready at %s.", self.directory)
                return
            except Exception as exc:
                logger.warning("Research memory ChromaDB unavailable (%s); using JSONL store.", exc)
        self._load_jsonl()

    def _jsonl_path(self) -> Path:
        return self.directory / _JSONL_FILENAME

    def _load_jsonl(self) -> None:
        try:
            with open(self._jsonl_path(), "r", encoding="utf-8") as f:
                self._jsonl_records = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            self._jsonl_records = []
        except (OSError, ValueError) as exc:
            logger.warning("Research memory JSONL unreadable (%s); starting empty.", exc)
            self._jsonl_records = []

    def _append_jsonl(self, records: list[dict[str, Any]]) -> None:
        with open(self._jsonl_path(), "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    # -- write ----------------------------------------------------------
    def save(
        self,
        user_id: str,
        query: str,
        report: str,
        *,
        evidence_level: str = "",
        consultation_id: str = "",
    ) -> int:
        """Ingest one provider-synthesised report. Returns chunks stored (0 skips)."""
        user_id = (user_id or "").strip()
        report = (report or "").strip()
        if not user_id or user_id == _ANONYMOUS_USER or not report:
            return 0
        chunks = _chunk_text(report)
        if not chunks:
            return 0
        created_at = _utcnow()
        base_id = (consultation_id or "").strip() or uuid.uuid4().hex
        with self._lock:
            try:
                if self._chroma_collection is not None:
                    self._save_chroma(user_id, query, chunks, evidence_level, created_at, base_id)
                else:
                    self._save_jsonl(user_id, query, chunks, evidence_level, created_at, base_id)
            except Exception as exc:
                logger.warning("Research memory save failed: %s", exc)
                return 0
        return len(chunks)

    def _save_chroma(self, user_id, query, chunks, evidence_level, created_at, base_id) -> None:
        ids = [f"{base_id}:{i}" for i in range(len(chunks))]
        query_note = (query or "").strip()[:200]
        metadatas = [
            {"user_id": user_id, "query": query_note, "evidence_level": evidence_level or "unknown", "created_at": created_at}
            for _ in chunks
        ]
        self._chroma_collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        self._prune_chroma(user_id)

    def _prune_chroma(self, user_id: str) -> None:
        stored = self._chroma_collection.get(where={"user_id": user_id}, include=["metadatas"])
        ids = stored.get("ids") or []
        if len(ids) <= _MAX_CHUNKS_PER_USER:
            return
        metas = stored.get("metadatas") or []
        ranked = sorted(zip(ids, metas), key=lambda pair: str((pair[1] or {}).get("created_at", "")))
        drop = [doc_id for doc_id, _ in ranked[: len(ids) - _MAX_CHUNKS_PER_USER]]
        if drop:
            self._chroma_collection.delete(ids=drop)

    def _save_jsonl(self, user_id, query, chunks, evidence_level, created_at, base_id) -> None:
        records = [
            {
                "id": f"{base_id}:{i}",
                "user_id": user_id,
                "query": (query or "").strip()[:200],
                "evidence_level": evidence_level or "unknown",
                "created_at": created_at,
                "content": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]
        self._append_jsonl(records)
        self._jsonl_records.extend(records)
        self._engine_size = -1  # force TF-IDF rebuild on next query
        # Prune oldest chunks beyond the per-user cap.
        own = [r for r in self._jsonl_records if r.get("user_id") == user_id]
        if len(own) > _MAX_CHUNKS_PER_USER:
            # Oldest first, so the newest chunks survive the cut.
            by_age = sorted(own, key=lambda r: str(r.get("created_at", "")))
            excess = len(own) - _MAX_CHUNKS_PER_USER
            drop_ids = {r["id"] for r in by_age[:excess]}
            self._jsonl_records = [r for r in self._jsonl_records if r.get("id") not in drop_ids]
            self._rewrite_jsonl()

    def _rewrite_jsonl(self) -> None:
        tmp_path = self._jsonl_path().with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in self._jsonl_records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp_path, self._jsonl_path())

    # -- read -----------------------------------------------------------
    def query(self, user_id: str, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to ``top_k`` cached chunks for this user only (never raises)."""
        user_id = (user_id or "").strip()
        if not user_id or user_id == _ANONYMOUS_USER or not (query_text or "").strip():
            return []
        with self._lock:
            try:
                if self._chroma_collection is not None:
                    return self._query_chroma(user_id, query_text, top_k)
                return self._query_jsonl(user_id, query_text, top_k)
            except Exception as exc:
                logger.warning("Research memory query failed: %s", exc)
                return []

    def _query_chroma(self, user_id: str, query_text: str, top_k: int) -> list[dict[str, Any]]:
        res = self._chroma_collection.query(
            query_texts=[query_text], n_results=top_k, where={"user_id": user_id}
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        hits = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            hits.append(
                {
                    "content": doc,
                    "query": meta.get("query", ""),
                    "evidence_level": meta.get("evidence_level", "unknown"),
                    "created_at": meta.get("created_at", ""),
                }
            )
        return hits

    def _query_jsonl(self, user_id: str, query_text: str, top_k: int) -> list[dict[str, Any]]:
        own = [r for r in self._jsonl_records if r.get("user_id") == user_id]
        if not own:
            return []
        if self._engine is None or self._engine_size != len(self._jsonl_records):
            from common.dermatology_rag import FallbackVectorEngine

            docs = []
            for r in self._jsonl_records:
                docs.append(
                    {
                        "id": r["id"],
                        "title": r.get("query", ""),
                        "category": r.get("evidence_level", ""),
                        "content": r.get("content", ""),
                    }
                )
            self._engine = FallbackVectorEngine(docs)
            self._engine_size = len(self._jsonl_records)
        # Include each record's original question so topic words help matching.
        past_questions = " ".join(r.get("query", "") for r in own)
        scored = self._engine.search(f"{query_text} {past_questions}", top_k=top_k * 3)
        by_id = {r["id"]: r for r in own}
        hits = []
        for doc in scored:
            record = by_id.get(doc.get("id"))
            if record is None or len(hits) >= top_k:
                continue
            hits.append(
                {
                    "content": record.get("content", ""),
                    "query": record.get("query", ""),
                    "evidence_level": record.get("evidence_level", "unknown"),
                    "created_at": record.get("created_at", ""),
                }
            )
        return hits


# Singleton keyed by env config so tests can point it at a tmp dir.
_memory_instance: ResearchMemory | None = None
_memory_key: tuple[str, str] | None = None
_memory_lock = threading.Lock()


def get_research_memory() -> ResearchMemory:
    """Return the shared memory store, rebuilding it if env config changed."""
    global _memory_instance, _memory_key
    key = (str(_memory_dir()), _backend())
    with _memory_lock:
        if _memory_instance is None or _memory_key != key:
            _memory_instance = ResearchMemory()
            _memory_key = key
        return _memory_instance


def reset_research_memory() -> None:
    """Drop the singleton (tests only)."""
    global _memory_instance, _memory_key
    with _memory_lock:
        _memory_instance = None
        _memory_key = None


def save_research_memory(
    user_id: str,
    query: str,
    report: str,
    *,
    evidence_level: str = "",
    consultation_id: str = "",
    report_generated_by: str = "",
) -> int:
    """Ingest a research report into the vector DB.

    Only provider-synthesised reports are stored — local fallbacks would
    pollute the index with generic text. Never raises.
    """
    try:
        if report_generated_by and report_generated_by != "provider":
            return 0
        return get_research_memory().save(
            user_id, query, report, evidence_level=evidence_level, consultation_id=consultation_id
        )
    except Exception as exc:
        logger.warning("Research memory save failed: %s", exc)
        return 0


def query_research_memory(user_id: str, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Read cached research for this user only. Never raises (``[]`` on failure)."""
    try:
        return get_research_memory().query(user_id, query_text, top_k=top_k)
    except Exception as exc:
        logger.warning("Research memory query failed: %s", exc)
        return []
