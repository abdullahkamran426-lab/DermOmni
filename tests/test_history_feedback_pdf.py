"""Tests for Consultation History, PDF Export, and Feedback/Eval Loop.

Offline: uses tmp HISTORY_DB_PATH + mocked brains; never hits live providers.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "test-history.db"
    monkeypatch.setenv("HISTORY_DB_PATH", str(db))
    import importlib

    import common.history_store as store

    importlib.reload(store)
    store.init_db()
    # Rebind main's reference to the reloaded module.
    import main as app_module

    app_module.history_store = store
    return store


def test_store_roundtrip(isolated_db):
    store = isolated_db
    saved = store.save_consultation(
        user_id="user-1", kind="consult",
        input_text="itchy patch", transcript="itchy patch",
        guidance="Keep clean. See a dermatologist.",
        media_kind="none", latency_ms=120, model_meta={"consult_model": "test"},
    )
    assert saved["id"]
    items, total = store.list_consultations("user-1")
    assert total == 1 and items[0]["id"] == saved["id"]
    # Isolation: other users see nothing.
    items2, total2 = store.list_consultations("user-2")
    assert total2 == 0
    fetched = store.get_consultation(saved["id"], "user-1")
    assert fetched and fetched["guidance"].startswith("Keep clean")
    assert store.get_consultation(saved["id"], "user-2") is None
    assert store.delete_consultation(saved["id"], "user-2") is False
    assert store.delete_consultation(saved["id"], "user-1") is True


def test_feedback_and_eval_export(isolated_db):
    store = isolated_db
    saved = store.save_consultation(user_id="u1", kind="research",
                                    input_text="q", report="VISUAL OBSERVATIONS\nok")
    fb = store.save_feedback(consultation_id=saved["id"], user_id="u1",
                             rating="1", note="helpful", run_metadata={"kind": "research"})
    assert fb["rating"] == "up"
    with pytest.raises(ValueError):
        store.save_feedback(consultation_id=saved["id"], user_id="u1", rating="meh")
    with pytest.raises(KeyError):
        store.save_feedback(consultation_id="nope", user_id="u1", rating="up")
    items, total = store.list_feedback(consultation_id=saved["id"])
    assert total == 1
    body = store.export_eval_jsonl()
    assert saved["id"] in body and '"rating": "up"' in body


def test_pdf_builder_smoke():
    from common.pdf_report import build_consultation_pdf, parse_report_sections

    report = ("VISUAL OBSERVATIONS\nRed patch noted.\n\nPOTENTIAL CONDITIONS\n"
              "Could be irritation.\n\nRECOMMENDATIONS\nKeep clean.\n\n"
              "WHEN TO SEE A DOCTOR\nIf spreading.")
    assert len(parse_report_sections(report)) == 4
    pdf = build_consultation_pdf({
        "id": "abc123", "kind": "research", "created_at": "2026-09-14T00:00:00+00:00",
        "input_text": "red patch", "visual_analysis": "red patch noted",
        "report": report, "guidance": "",
        "evidence": {"evidence_level": "moderate", "source_count": 2},
        "sources": [{"title": "AAD", "url": "https://www.aad.org/x"}],
        "media_kind": "image",
    })
    assert pdf[:5] == b"%PDF-"


def test_analyze_persists_history(client, mock_brain, mock_tts, isolated_db):
    resp = client.post("/api/analyze",
                       data={"text": "Itchy red patch.", "user_id": "hist-user-1"})
    assert resp.status_code == 200
    payload = resp.json()
    assert "consultation_id" in payload
    hist = client.get("/api/history", params={"user_id": "hist-user-1"})
    assert hist.status_code == 200
    assert hist.json()["total"] == 1
    # Other user cannot see it.
    hist2 = client.get("/api/history", params={"user_id": "someone-else"})
    assert hist2.json()["total"] == 0
    # PDF export works scoped to owner.
    pdf = client.get(f"/api/history/{payload['consultation_id']}/export.pdf",
                     params={"user_id": "hist-user-1"})
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:5] == b"%PDF-"
    # Cross-user PDF is 404.
    denied = client.get(f"/api/history/{payload['consultation_id']}/export.pdf",
                        params={"user_id": "intruder"})
    assert denied.status_code == 404


def test_photo_archive_and_pdf_embed(isolated_db, tmp_path):
    """Uploaded still image is archived and embedded in the PDF."""
    from PIL import Image

    from common.pdf_report import build_consultation_pdf

    store = isolated_db
    src = tmp_path / "upload.jpg"
    Image.new("RGB", (200, 120), color="red").save(src, format="JPEG")

    base_record = {
        "id": "photoid", "kind": "consult", "created_at": "2026-09-14T00:00:00+00:00",
        "input_text": "red patch", "transcript": "", "guidance": "Keep clean.",
        "report": "", "visual_analysis": "", "evidence": {}, "sources": [],
        "media_kind": "image",
    }
    pdf_without = build_consultation_pdf({**base_record, "media_image_path": ""})

    saved = store.save_consultation(user_id="photo-user", kind="consult",
                                    input_text="red patch", guidance="Keep clean.",
                                    media_kind="image")
    archived = store.archive_consultation_image(saved["id"], src)
    assert archived and archived.endswith(".jpg")
    fetched = store.get_consultation(saved["id"], "photo-user")
    assert fetched["media_image_path"] == archived

    pdf_with = build_consultation_pdf(fetched)
    assert pdf_with[:5] == b"%PDF-"
    # Embedded JPEG must appear as an image XObject and grow the PDF.
    assert b"/XObject" in pdf_with or b"/Image" in pdf_with
    assert len(pdf_with) > len(pdf_without) + 500

    # Deleting the session removes the archived photo file.
    from pathlib import Path as _Path
    assert _Path(archived).is_file()
    assert store.delete_consultation(saved["id"], "photo-user") is True
    assert not _Path(archived).exists()


def test_analyze_with_image_archives_photo(client, mock_brain, mock_tts, isolated_db, sample_image_bytes):
    resp = client.post(
        "/api/analyze",
        data={"text": "Red patch.", "user_id": "img-user"},
        files={"image": ("skin.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    cid = resp.json()["consultation_id"]
    record = client.get("/api/history/" + cid, params={"user_id": "img-user"}).json()
    assert record["media_image_path"], "expected archived photo path"
    pdf = client.get(f"/api/history/{cid}/export.pdf", params={"user_id": "img-user"})
    assert pdf.status_code == 200
    assert pdf.content[:5] == b"%PDF-"


def test_feedback_api_flow(client, mock_brain, mock_tts, isolated_db):
    resp = client.post("/api/analyze",
                       data={"text": "Bump that itches.", "user_id": "fb-user"},
                       headers={"X-User-Id": "fb-user"})
    cid = resp.json()["consultation_id"]
    fb = client.post("/api/feedback", json={
        "consultation_id": cid, "rating": "down",
        "note": "too generic", "user_id": "fb-user"})
    assert fb.status_code == 201
    assert fb.json()["rating"] == "down"
    assert fb.json()["run_metadata"]["kind"] == "consult"
    bad = client.post("/api/feedback", json={
        "consultation_id": cid, "rating": "meh", "user_id": "fb-user"})
    assert bad.status_code == 400
    missing = client.post("/api/feedback", json={
        "consultation_id": "deadbeef", "rating": "up", "user_id": "fb-user"})
    assert missing.status_code == 404
    evals = client.get("/api/evals/export")
    assert evals.status_code == 200
    assert cid in evals.text
