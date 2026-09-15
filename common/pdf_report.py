"""Clinician-ready PDF export (ReportLab Platypus).

Pattern: pure function ``build_consultation_pdf(record: dict) -> bytes``.
The FastAPI layer fetches a history row (ownership-checked) and streams the
bytes; no temp files, no global state — trivially unit-testable.

Layout:
  header brand bar -> meta table -> disclaimer callout -> uploaded photo
  (if the session has an archived still image) -> content sections
  -> sources (numbered, URL in muted mono) -> evidence box -> footer.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SAGE = colors.HexColor("#1F6F5C")
SAGE_SOFT = colors.HexColor("#E3EEE8")
INK = colors.HexColor("#16231F")
MUTED = colors.HexColor("#5B655F")
BORDER = colors.HexColor("#E1DFD1")
AMBER = colors.HexColor("#C98A2C")
AMBER_SOFT = colors.HexColor("#F5E6C9")

DISCLAIMER_TEXT = (
    "For general information only — not a diagnosis and not a substitute for "
    "care from a licensed dermatologist or clinician. Seek prompt in-person "
    "care for severe pain, rapidly spreading redness, fever, pus, blistering, "
    "or symptoms that persist or worsen."
)

_SECTION_HEADINGS = (
    "VISUAL OBSERVATIONS",
    "POTENTIAL CONDITIONS",
    "RECOMMENDATIONS",
    "WHEN TO SEE A DOCTOR",
)


def parse_report_sections(report: str) -> list[tuple[str, str]]:
    """Split a 4-section synthesis report into (heading, body) pairs.

    Falls back to a single 'CLINICAL GUIDANCE' section for consult outputs.
    """
    text = (report or "").strip()
    if not text:
        return []
    pattern = "(" + "|".join(_SECTION_HEADINGS) + r")\s*\n"
    parts = re.split(pattern, text)
    # re.split with capture group -> [pre, H1, B1, H2, B2, ...]
    sections: list[tuple[str, str]] = []
    if len(parts) >= 3:
        pre = parts[0].strip()
        if pre:
            sections.append(("OVERVIEW", pre))
        for i in range(1, len(parts) - 1, 2):
            heading = parts[i].strip().upper()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if body:
                sections.append((heading, body))
        if sections:
            return sections
    # Consult-style free text (no canonical headings).
    return [("CLINICAL GUIDANCE", text)]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title2", parent=base["Title"], fontSize=20,
                                leading=24, textColor=INK, spaceAfter=2),
        "subtitle": ParagraphStyle("Sub", parent=base["Normal"], fontSize=9,
                                   leading=12, textColor=MUTED),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=12,
                             leading=15, textColor=SAGE, spaceBefore=14,
                             spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("Body2", parent=base["Normal"], fontSize=10,
                               leading=15, textColor=INK, spaceAfter=6),
        "callout": ParagraphStyle("Callout", parent=base["Normal"], fontSize=9,
                                  leading=13, textColor=INK),
        "muted": ParagraphStyle("Muted", parent=base["Normal"], fontSize=8.5,
                                leading=11.5, textColor=MUTED),
        "mono": ParagraphStyle("Mono", parent=base["Normal"], fontSize=8,
                               leading=11, textColor=MUTED, fontName="Courier"),
        "footer": ParagraphStyle("Footer", parent=base["Normal"], fontSize=8,
                                 leading=10, textColor=MUTED, alignment=1),
    }


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(
        A4[0] / 2, 15 * mm,
        f"AI Skin Specialist — informational only  •  Page {doc.page}",
    )
    canvas.restoreState()


# Max on-page size of the embedded patient photo.
_PHOTO_MAX_W = 150 * mm
_PHOTO_MAX_H = 90 * mm


def _find_embeddable_image(record: dict) -> Path | None:
    """Return the archived patient photo path when it exists and is readable."""
    candidate = (record or {}).get("media_image_path") or ""
    if not candidate:
        return None
    path = Path(candidate)
    try:
        if path.is_file() and path.stat().st_size > 0:
            # Validate it is a real image (corrupt files must not break PDFs).
            ImageReader(str(path))
            return path
    except Exception:
        return None
    return None


def _find_annotated_image(record: dict) -> Path | None:
    """Return the lesion-overlay image path when it exists and is readable."""
    candidate = (record or {}).get("annotated_image_path") or ""
    if not candidate:
        return None
    path = Path(candidate)
    try:
        if path.is_file() and path.stat().st_size > 0:
            ImageReader(str(path))
            return path
    except Exception:
        return None
    return None


def _scaled_photo_flowable(path: Path):
    """Build a aspect-preserved ReportLab Image flowable, or None on failure."""
    try:
        reader = ImageReader(str(path))
        iw, ih = reader.getSize()
        if not iw or not ih:
            return None
        scale = min(_PHOTO_MAX_W / iw, _PHOTO_MAX_H / ih, 1.0)
        return RLImage(str(path), width=iw * scale, height=ih * scale)
    except Exception:
        return None


def build_consultation_pdf(record: dict) -> bytes:
    """Render a stored consultation/research row to PDF bytes."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title=f"Skin consultation {record.get('id', '')[:8]}",
        author="AI Skin Specialist",
    )

    kind = record.get("kind", "consult")
    created = record.get("created_at", "")
    try:
        created_fmt = datetime.fromisoformat(str(created)).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        created_fmt = str(created or "—")

    story = []
    story.append(Paragraph("AI Skin Specialist", styles["title"]))
    story.append(Paragraph(
        f"{'Evidence-based research report' if kind == 'research' else 'Consultation summary'}"
        f" &nbsp;•&nbsp; {created_fmt} &nbsp;•&nbsp; ID {str(record.get('id', ''))[:8]}",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=SAGE, spaceAfter=8))

    # Meta table: input type / media / evidence.
    evidence = record.get("evidence") or {}
    meta_rows = [
        [Paragraph("<b>Session kind</b>", styles["muted"]),
         Paragraph(kind, styles["body"]),
         Paragraph("<b>Media</b>", styles["muted"]),
         Paragraph(str(record.get("media_kind", "none")), styles["body"])],
        [Paragraph("<b>Evidence</b>", styles["muted"]),
         Paragraph(str(evidence.get("evidence_level", "—")), styles["body"]),
         Paragraph("<b>Sources</b>", styles["muted"]),
         Paragraph(str(evidence.get("source_count", len(record.get("sources") or []))), styles["body"])],
    ]
    meta = Table(meta_rows, colWidths=[28 * mm, 45 * mm, 28 * mm, 45 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SAGE_SOFT),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta)
    story.append(Spacer(1, 8))

    # Disclaimer callout.
    callout = Table([[Paragraph(f"<b>Medical disclaimer.</b> {DISCLAIMER_TEXT}", styles["callout"])]],
                    colWidths=[174 * mm])
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER_SOFT),
        ("BOX", (0, 0), (-1, -1), 0.75, AMBER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(callout)

    def _escape(text: str) -> str:
        """Escape a plain-text value so it is safe inside a ReportLab Paragraph."""
        escaped = text.replace("&", "&amp;")
        escaped = escaped.replace("<", "&lt;")
        escaped = escaped.replace(">", "&gt;")
        escaped = escaped.replace("\n", "<br/>")
        return escaped

    # Uploaded patient photo (archived still image for this session).
    photo_path = _find_embeddable_image(record)
    if photo_path is not None:
        photo = _scaled_photo_flowable(photo_path)
        if photo is not None:
            story.append(Paragraph("Uploaded photo", styles["h2"]))
            story.append(photo)
            story.append(Paragraph(
                "Photo provided by the patient at consultation time. "
                "Video/audio are never stored; only this still image is archived.",
                styles["muted"],
            ))
            story.append(Spacer(1, 4))

    # Lesion-overlay annotation (Gemini bounding boxes + Pillow callouts).
    annotated_path = _find_annotated_image(record)
    if annotated_path is not None:
        annotated_photo = _scaled_photo_flowable(annotated_path)
        if annotated_photo is not None:
            story.append(Paragraph("Annotated analysis overlay", styles["h2"]))
            story.append(annotated_photo)
            story.append(Paragraph(
                "AI-generated lesion callouts and risk highlights "
                "(green = mild, amber = moderate, red = severe). "
                "For visual reference only — not a diagnosis.",
                styles["muted"],
            ))
            story.append(Spacer(1, 4))

    # Patient input / transcript.
    patient_text = record.get("input_text") or record.get("transcript") or ""
    if patient_text:
        story.append(Paragraph("Patient description", styles["h2"]))
        story.append(Paragraph(_escape(str(patient_text)[:3000]), styles["body"]))

    visual = (record.get("visual_analysis") or "").strip()
    main_text = (record.get("report") or record.get("guidance") or "").strip()
    if kind == "research":
        if visual:
            story.append(Paragraph("Visual analysis (on file)", styles["h2"]))
            story.append(Paragraph(_escape(visual[:3000]), styles["body"]))
        for heading, body in parse_report_sections(main_text):
            story.append(Paragraph(heading.title(), styles["h2"]))
            story.append(Paragraph(_escape(body[:6000]), styles["body"]))
    else:
        if main_text:
            story.append(Paragraph("Doctor's guidance", styles["h2"]))
            story.append(Paragraph(_escape(main_text[:6000]), styles["body"]))
        if visual:
            story.append(Paragraph("Visual findings", styles["h2"]))
            story.append(Paragraph(_escape(visual[:3000]), styles["body"]))

    # Sources.
    sources = record.get("sources") or []
    if sources:
        story.append(Paragraph("Sources & further reading", styles["h2"]))
        items = []
        for src in sources[:20]:
            title = _escape(str(src.get("title", "Untitled"))[:200])
            url = _escape(str(src.get("url", ""))[:300])
            items.append(ListItem(
                [Paragraph(f"{title}<br/><font color=\"#5B655F\">{url}</font>", styles["body"])],
                leftIndent=14, spaceBefore=2,
            ))
        story.append(ListFlowable(items, bulletType="1", start="1"))
    else:
        story.append(Paragraph("Sources & further reading", styles["h2"]))
        story.append(Paragraph("No external sources were attached to this session.", styles["muted"]))

    # Evidence box.
    if evidence:
        story.append(Paragraph("Evidence notes", styles["h2"]))
        ev_rows = [
            [Paragraph(f"<b>{k.replace('_', ' ')}</b>", styles["muted"]),
             Paragraph(_escape(str(v)[:500]), styles["body"])]
            for k, v in evidence.items()
        ]
        ev_table = Table(ev_rows, colWidths=[45 * mm, 129 * mm])
        ev_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(ev_table)

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))
    story.append(Paragraph(
        "Generated by AI Skin Specialist for educational purposes. "
        "Only the still photo above (when provided) is archived; "
        "audio/video are never stored.",
        styles["footer"],
    ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
