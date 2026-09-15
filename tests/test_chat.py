"""Tests for POST /api/chat (history-grounded follow-up Q&A).

Offline: the Gemini call is mocked via main._run_chat_reply; history rows are
created through the real /api/analyze path with mocked brain/TTS.
"""
from __future__ import annotations


def _make_session(client, mock_brain, mock_tts, user_id="chat-user"):
    resp = client.post(
        "/api/analyze",
        data={"text": "Red itchy patch on forearm.", "user_id": user_id},
        headers={"X-User-Id": user_id},
    )
    assert resp.status_code == 200
    return resp.json()["consultation_id"]


def test_chat_grounded_in_session(client, mock_brain, mock_tts, mocker):
    cid = _make_session(client, mock_brain, mock_tts)
    mock_reply = mocker.patch(
        "main._run_chat_reply",
        return_value="Grounded reply. Keep the area clean and see a dermatologist if it spreads.",
    )
    resp = client.post(
        "/api/chat",
        json={"message": "Should I cover it?", "consultation_id": cid, "user_id": "chat-user"},
        headers={"X-User-Id": "chat-user"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["consultation_id"] == cid
    assert payload["reply"].startswith("Grounded reply")
    # Prompt carries the stored session context.
    prompt = mock_reply.call_args[0][0]
    assert "Red itchy patch" in prompt
    assert "Should I cover it?" in prompt


def test_chat_general_mode_without_session(client, mocker):
    mocker.patch("main._run_chat_reply", return_value="General reply.")
    resp = client.post("/api/chat", json={"message": "What is moisturizer?"})
    assert resp.status_code == 200
    assert resp.json()["consultation_id"] is None


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_unknown_session_404(client):
    resp = client.post(
        "/api/chat",
        json={"message": "Hi?", "consultation_id": "deadbeef", "user_id": "ghost"},
    )
    assert resp.status_code == 404


def test_chat_cross_user_session_404(client, mock_brain, mock_tts, mocker):
    cid = _make_session(client, mock_brain, mock_tts, user_id="owner-1")
    mocker.patch("main._run_chat_reply", return_value="Should never surface.")
    resp = client.post(
        "/api/chat",
        json={"message": "Hi?", "consultation_id": cid, "user_id": "intruder"},
        headers={"X-User-Id": "intruder"},
    )
    assert resp.status_code == 404


def test_chat_provider_failure_502(client, mock_brain, mock_tts, mocker):
    from common.errors import GuidanceError

    cid = _make_session(client, mock_brain, mock_tts, user_id="chat-user-502")
    mocker.patch(
        "main._run_chat_reply",
        side_effect=GuidanceError("boom", code="guidance_request_failed"),
    )
    resp = client.post(
        "/api/chat",
        json={"message": "Hi?", "consultation_id": cid, "user_id": "chat-user-502"},
    )
    assert resp.status_code == 502
