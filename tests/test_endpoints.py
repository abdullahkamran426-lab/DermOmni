"""Endpoint integration tests for POST /api/analyze.

All provider calls (Gemini brain, Deepgram TTS/STT) are mocked via the
`mock_brain` / `mock_tts` fixtures in conftest.py — these tests run 100%
offline and never touch live API quotas.
"""
from __future__ import annotations

import httpx


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_check_async():
    """Same health check over a real async httpx transport (pytest-asyncio)."""
    import main as app_module
    from httpx import ASGITransport

    transport = ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as async_client:
        response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_valid_text(client, mock_brain, mock_tts):
    response = client.post(
        "/api/analyze", data={"text": "Itchy red patch on my forearm."}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "Itchy red patch on my forearm."
    assert "guidance" in payload and payload["guidance"]
    assert payload["audio_url"].startswith("/audio/")
    mock_brain.assert_called_once()
    mock_tts.assert_called_once()


def test_analyze_valid_image_only(client, mock_brain, mock_tts, sample_image_bytes):
    response = client.post(
        "/api/analyze",
        files={"image": ("skin.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    # Image-only calls fall back to a default brain prompt; transcript is empty.
    assert payload["transcript"] == ""
    assert "guidance" in payload and payload["guidance"]
    assert payload["audio_url"].startswith("/audio/")
    mock_brain.assert_called_once()
    mock_tts.assert_called_once()


def test_analyze_combined_text_and_image(
    client, mock_brain, mock_tts, sample_image_bytes
):
    response = client.post(
        "/api/analyze",
        data={"text": "Raised bump that itches at night."},
        files={"image": ("skin.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "Raised bump that itches at night."
    assert "guidance" in payload and payload["guidance"]
    assert payload["audio_url"].startswith("/audio/")
    mock_brain.assert_called_once()
    mock_tts.assert_called_once()


def test_analyze_missing_input_returns_400(client):
    response = client.post("/api/analyze", data={"text": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_analyze_rate_limit_returns_429(client, mock_brain, mock_tts):
    """Exceed the 5/minute slowapi limit and expect HTTP 429."""
    statuses: list[int] = []
    for _ in range(6):
        response = client.post(
            "/api/analyze", data={"text": "Rate limit probe query."}
        )
        statuses.append(response.status_code)

    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
    assert "detail" in response.json()
