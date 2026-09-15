"""Unit tests for media preparation + inline-vs-File-API routing.

- `common/media.py`: small image -> JPEG bytes, video -> raw bytes + MIME,
  undecodable file -> VisionAnalysisError (all offline via Pillow/tmp files).
- `common/gemini.needs_file_api`: size/video-threshold routing.
- `Skin_research_tools._run_vision_analysis`: small inputs use inline byte
  parts (`models.generate_content`); large/video inputs route through
  `client.files.upload`. Gemini/Tavily clients are mocked throughout.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.errors import VisionAnalysisError
from common.gemini import needs_file_api
from common.media import detect_media_type, prepare_vision_media


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "skin.jpg"
    Image.new("RGB", (100, 100), color="red").save(path, format="JPEG")
    return path


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "lesion.mp4"
    path.write_bytes(b"\x00\x01\x02" * 1024)  # 3 KiB — content never decoded
    return path


# --- common/media.py -------------------------------------------------------

def test_detect_media_type_image_vs_video(tmp_path: Path):
    assert detect_media_type(tmp_path / "a.jpg") == "image"
    assert detect_media_type(tmp_path / "a.mp4") == "video"
    assert detect_media_type(tmp_path / "a.WEBM") == "video"


def test_prepare_small_image_returns_jpeg_bytes(small_image: Path):
    media_bytes, mime_type, media_type = prepare_vision_media(small_image)
    assert media_type == "image"
    assert mime_type == "image/jpeg"
    assert isinstance(media_bytes, bytes) and len(media_bytes) > 0


def test_prepare_video_returns_raw_bytes_with_mime(tiny_video: Path):
    media_bytes, mime_type, media_type = prepare_vision_media(tiny_video)
    assert media_type == "video"
    assert mime_type == "video/mp4"
    assert media_bytes == tiny_video.read_bytes()


def test_prepare_undecodable_image_raises(tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"this is not an image at all")
    with pytest.raises(VisionAnalysisError) as exc_info:
        prepare_vision_media(bad)
    assert exc_info.value.code == "invalid_media"


# --- common/gemini.needs_file_api ------------------------------------------

def test_small_image_does_not_need_file_api(small_image: Path):
    assert needs_file_api(small_image) is False


def test_video_always_needs_file_api(tiny_video: Path):
    assert needs_file_api(tiny_video) is True


def test_large_file_needs_file_api(small_image: Path, monkeypatch):
    monkeypatch.setattr("common.gemini.FILE_API_THRESHOLD_BYTES", 10)
    assert needs_file_api(small_image) is True


# --- Skin_research_tools._run_vision_analysis routing -----------------------

def _fake_rotation(return_value):
    """Replacement for call_with_key_rotation: runs the op on a Mock client.

    The real client factory would build a live `genai.Client`, so it is
    deliberately *not* invoked — the Mock stands in for the Gemini client
    (offline) while preserving the inline-vs-File-API call routing.
    """
    state: dict = {}

    def _fake(operation, client_factory, **kwargs):
        client = Mock(name="gemini-client")
        client.models.generate_content.return_value = return_value
        state["client"] = client
        return operation(client)

    def _get_mock_client():
        return state["client"]

    _fake.mock_client = _get_mock_client
    return _fake


def test_small_input_uses_inline_byte_parts(mocker, small_image: Path):
    import Skin_research_tools as tools

    fake_response = SimpleNamespace(candidates=[], text="inline visual analysis")
    fake_rotation = mocker.patch.object(
        tools, "call_with_key_rotation", side_effect=_fake_rotation(fake_response)
    )
    mocker.patch.object(
        tools,
        "prepare_vision_media",
        return_value=(b"fake-jpeg-bytes", "image/jpeg", "image"),
    )
    upload_spy = mocker.patch.object(tools, "upload_file_and_wait")

    result = tools._run_vision_analysis(small_image, "describe this", "itchy patch")

    assert result == "inline visual analysis"
    assert fake_rotation.call_count == 1
    # Inline path streams bytes via generate_content — never touches File API.
    mock_client = fake_rotation.side_effect.mock_client()
    assert mock_client.models.generate_content.call_count == 1
    mock_client.files.upload.assert_not_called()
    upload_spy.assert_not_called()


def test_large_input_routes_through_files_upload(mocker, tiny_video: Path):
    import Skin_research_tools as tools

    uploaded = SimpleNamespace(name="files/abc123", state=None)
    fake_response = SimpleNamespace(candidates=[], text="file-api visual analysis")

    mocker.patch.object(tools, "needs_file_api", return_value=True)
    upload_mock = mocker.patch.object(tools, "upload_file_and_wait", return_value=uploaded)
    delete_mock = mocker.patch.object(tools, "delete_remote_file")
    fake_rotation = mocker.patch.object(
        tools, "call_with_key_rotation", side_effect=_fake_rotation(fake_response)
    )

    result = tools._run_vision_analysis(tiny_video, "describe this video", "spreading rash")

    assert result == "file-api visual analysis"
    assert fake_rotation.call_count == 1
    # File API path uploads from disk, then cleans up the remote object.
    assert upload_mock.call_count == 1
    upload_mock.assert_called_once()
    mock_client = fake_rotation.side_effect.mock_client()
    assert mock_client.models.generate_content.call_count == 1
    assert delete_mock.call_count == 1
