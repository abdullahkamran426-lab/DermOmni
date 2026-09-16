"""
Multimodal Image Annotation module for DermOmni.
Uses Gemini Vision to detect skin abnormalities and Pillow to draw bounding box callouts.
"""

import io
import os
import json
import uuid
import logging
from pathlib import Path as _Path
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw

from google import genai
from google.genai import types
from common.gemini import call_with_key_rotation, get_gemini_api_keys

logger = logging.getLogger(__name__)

import tempfile

if os.environ.get("VERCEL"):
    ANNOTATED_OUTPUT_DIR = str(_Path(tempfile.gettempdir()) / "generated_audio")
else:
    ANNOTATED_OUTPUT_DIR = str(_Path(__file__).resolve().parent.parent / "generated_audio")
_Path(ANNOTATED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)



def detect_skin_lesion_regions(image_bytes: bytes, mime_type: str = "image/jpeg") -> List[Dict[str, Any]]:
    """
    Use Gemini 2.5 Flash to identify bounding boxes for skin lesion features.
    Returns a list of dicts: [{"label": "Erythema", "box_2d": [ymin, xmin, ymax, xmax], "confidence": "high", "severity": "mild"}]
    Coordinates normalized to [0, 1000] scale.
    """
    prompt = """
Analyze this clinical skin image and locate key visible skin abnormalities or features of interest (e.g. Primary lesion, Erythema, Hyperpigmentation, Irregular Border, Scaling, Normal skin).
For each detected feature, return a JSON array of objects with:
- "label": Short descriptive label (e.g. "Primary Lesion", "Erythematous Border", "Hyperpigmented Core")
- "box_2d": Normalized bounding box coordinates as [ymin, xmin, ymax, xmax] on a 0 to 1000 scale.
- "severity": "mild", "moderate", or "severe"

Return ONLY a valid JSON array of objects, e.g.:
[
  {"label": "Primary Lesion", "box_2d": [250, 300, 650, 700], "severity": "moderate"}
]
"""
    try:
        get_gemini_api_keys()
        model_name = os.getenv("GEMINI_CONSULT_MODEL", "gemini-3.6-flash")
        def _op(client: genai.Client):
            res = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            return res.text

        response_text = call_with_key_rotation(_op, lambda k: genai.Client(api_key=k), operation_name="annotate-lesion")

        # Parse JSON output from Gemini
        clean_text = response_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        regions = json.loads(clean_text)
        if isinstance(regions, list) and len(regions) > 0:
            return regions
    except Exception as e:
        logger.warning(f"Gemini vision region detection failed or returned invalid JSON: {e}. Using heuristic fallbacks.")

    # Fallback default region if AI detection unavailable/failed
    return [
        {
            "label": "Region of Interest",
            "box_2d": [200, 200, 800, 800],
            "severity": "moderate"
        }
    ]


def annotate_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    output_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Draw visual bounding boxes and callout labels on a skin lesion image.
    Saves the image to ANNOTATED_OUTPUT_DIR and returns metadata including the file path/URL.
    """
    os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)

    # Load original image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    width, height = image.size

    # Detect regions using Gemini
    regions = detect_skin_lesion_regions(image_bytes, mime_type)

    # Create an overlay layer for translucent fill and sharp line drawing
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Color palette for severities
    color_map = {
        "mild": {"stroke": (34, 197, 94, 255), "fill": (34, 197, 94, 40)},       # Green
        "moderate": {"stroke": (245, 158, 11, 255), "fill": (245, 158, 11, 40)},  # Amber
        "severe": {"stroke": (239, 68, 68, 255), "fill": (239, 68, 68, 50)},      # Red
    }
    default_colors = {"stroke": (14, 165, 233, 255), "fill": (14, 165, 233, 40)}  # Cyan

    annotated_regions = []

    for region in regions:
        box_2d = region.get("box_2d", [200, 200, 800, 800])
        label = region.get("label", "Lesion Area")
        severity = region.get("severity", "moderate").lower()

        colors = color_map.get(severity, default_colors)

        # Convert 0-1000 scale to pixel coordinates
        ymin, xmin, ymax, xmax = box_2d
        left = int((xmin / 1000.0) * width)
        top = int((ymin / 1000.0) * height)
        right = int((xmax / 1000.0) * width)
        bottom = int((ymax / 1000.0) * height)

        # Clamp bounds
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 10, min(right, width))
        bottom = max(top + 10, min(bottom, height))

        # Draw box fill & thick border
        line_width = max(3, int(min(width, height) * 0.006))
        draw.rectangle([left, top, right, bottom], fill=colors["fill"], outline=colors["stroke"], width=line_width)

        # Draw label header box
        label_text = f" {label} ({severity.upper()}) "
        text_bbox = draw.textbbox((left, max(0, top - 25)), label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1] + 6

        label_top = max(0, top - text_height)
        draw.rectangle([left, label_top, left + text_width, label_top + text_height], fill=colors["stroke"])
        draw.text((left + 2, label_top + 2), label_text, fill=(255, 255, 255, 255))

        annotated_regions.append({
            "label": label,
            "severity": severity,
            "bounding_box_px": [left, top, right, bottom]
        })

    # Composite overlay onto original image
    final_image = Image.alpha_composite(image, overlay).convert("RGB")

    # Generate a unique filename per call — the old code derived the name
    # from the upload's basename (temp dirs reuse names like "image.jpg"),
    # so concurrent/second requests overwrote the first file and broke
    # previously returned /audio/ URLs.
    safe_suffix = uuid.uuid4().hex[:10]
    filename = f"annotated_{safe_suffix}.png"

    file_path = os.path.join(ANNOTATED_OUTPUT_DIR, filename)
    final_image.save(file_path, format="PNG")

    logger.info(f"Saved annotated image to {file_path}")

    return {
        "annotated_filename": filename,
        "annotated_image_url": f"/audio/{filename}",
        "file_path": file_path,
        "regions": annotated_regions
    }
