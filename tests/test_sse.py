"""
Integration tests for FastAPI SSE streaming endpoint (/api/analyze-stream).
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_analyze_stream_no_input():
    response = client.post("/api/analyze-stream", data={})
    assert response.status_code == 400


def test_analyze_stream_text_query(monkeypatch):
    """Test /api/analyze-stream event stream structure."""
    monkeypatch.setattr("main.brain_of_the_doctor", lambda patient_text, *args, **kw: "Doctor guidance test response.")
    monkeypatch.setattr("main.convert_text_to_doctor_audio", lambda text, output_filepath: open(output_filepath, "wb").write(b"fake mp3"))

    headers = {"Accept-Language": "es-ES,es;q=0.9"}
    response = client.post("/api/analyze-stream", data={"text": "Red rash on wrist"}, headers=headers)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "event: language_detected" in content
    assert '"lang": "es"' in content
    assert "event: stt_transcript" in content
    assert "Red rash on wrist" in content
    assert "event: rag_evidence" in content
    assert "event: guidance" in content
    assert "event: audio_url" in content
    assert "event: complete" in content
