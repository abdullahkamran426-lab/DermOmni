"""Shared pytest fixtures for the offline QA suite.

Guarantees:
- Dummy credentials are set before the app is imported (no real keys needed).
- Any direct construction of a live Gemini / Deepgram / Tavily client fails
  loudly, so tests can never hit live API quotas.
- The slowapi rate-limit storage is reset before/after every test so the
  5/minute limit on POST /api/analyze stays deterministic.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

# Dummy offline credentials — set before importing the application modules.
os.environ.setdefault("GEMINI_API_KEY", "test-key-1")
os.environ.setdefault("GEMINI_API_KEYS", "test-key-2")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("HF_API_KEY", "test-hf-key")

import pytest
from fastapi.testclient import TestClient

import main as app_module


def _blocked_client_factory(service_name: str):
    def _blocked(*args, **kwargs):
        raise AssertionError(
            f"Live {service_name} call blocked in tests (offline suite)."
        )

    return _blocked


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset slowapi in-memory storage so rate-limit tests are deterministic."""
    storage = getattr(app_module.app.state.limiter, "_storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
    yield
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()


@pytest.fixture(autouse=True)
def _block_live_clients(mocker):
    """Fail fast if any test path tries to build a real external client."""
    mocker.patch(
        "google.genai.Client",
        side_effect=_blocked_client_factory("Gemini"),
    )
    mocker.patch(
        "voice_of_the_patient.DeepgramClient",
        side_effect=_blocked_client_factory("Deepgram-STT"),
    )
    mocker.patch(
        "voice_of_the_doctor.DeepgramClient",
        side_effect=_blocked_client_factory("Deepgram-TTS"),
    )
    mocker.patch(
        "Skin_research_tools.TavilyClient",
        side_effect=_blocked_client_factory("Tavily"),
    )
    yield


@pytest.fixture()
def client():
    """Synchronous TestClient (httpx-based) bound to the FastAPI app."""
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture()
def mock_brain(mocker):
    """Mock the Gemini guidance step in main's namespace (no live calls)."""
    return mocker.patch(
        "main.brain_of_the_doctor",
        return_value=(
            "Test guidance. Keep the area clean and see a dermatologist "
            "if symptoms persist."
        ),
    )


@pytest.fixture()
def mock_tts(mocker):
    """Mock Deepgram TTS: write fake bytes to the expected output path."""

    def _fake_tts(text, output_filepath=None):
        path = Path(output_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"FAKE-MP3-BYTES")
        return path

    return mocker.patch(
        "main.convert_text_to_doctor_audio", side_effect=_fake_tts
    )


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """A small valid JPEG payload for image-upload tests."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()
