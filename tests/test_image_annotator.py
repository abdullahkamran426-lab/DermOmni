"""
Unit tests for common/image_annotator.py module.
"""

import os
import io
import pytest
from PIL import Image
from common.image_annotator import annotate_image, detect_skin_lesion_regions


@pytest.fixture
def sample_image_bytes():
    """Create a simple 200x200 red test image in PNG format."""
    img = Image.new("RGB", (200, 200), color=(220, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_skin_lesion_regions_fallback(sample_image_bytes, monkeypatch):
    """Test region detection with Gemini mocked to force fallback."""
    monkeypatch.setattr("common.image_annotator.get_gemini_api_keys", lambda: ["fake-key"])
    monkeypatch.setattr("common.image_annotator.call_with_key_rotation", lambda op, factory, operation_name: "invalid json response")
    regions = detect_skin_lesion_regions(sample_image_bytes, mime_type="image/png")
    assert isinstance(regions, list)
    assert len(regions) > 0
    assert "label" in regions[0]
    assert "box_2d" in regions[0]


def test_annotate_image_canvas_rendering(sample_image_bytes, monkeypatch):
    """Test full image annotation workflow producing output canvas."""
    fake_regions = [
        {"label": "Primary Lesion", "box_2d": [100, 100, 500, 500], "severity": "severe"}
    ]
    monkeypatch.setattr("common.image_annotator.detect_skin_lesion_regions", lambda img, mime: fake_regions)

    res = annotate_image(sample_image_bytes, mime_type="image/png", output_filename="test_skin.png")

    assert "annotated_image_url" in res
    assert "file_path" in res
    assert os.path.exists(res["file_path"])
    assert len(res["regions"]) == 1
    assert res["regions"][0]["label"] == "Primary Lesion"

    # Clean up generated file
    if os.path.exists(res["file_path"]):
        os.remove(res["file_path"])
