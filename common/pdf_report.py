"""Clinician-ready PDF export (ReportLab Platypus).

Pattern: pure function ``build_consultation_pdf(record: dict) -> bytes``.
The FastAPI layer fetches a history row (ownership-checked) and streams the
bytes; no temp files, no global state — trivially unit-testable.

Layout:
  header brand bar -> meta table -> disclaimer callout -> side-by-side image comparison
  (or single photo when only one image is available) -> content sections
  -> sources (numbered, URL in muted mono) -> evidence box -> running footer.
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

# Professional Healthcare & Editorial Color Palette
PRIMARY = colors.HexColor("#0F4C3A")       # Deep Emerald / Forest Teal
PRIMARY_SOFT = colors.HexColor("#EAF2EE")  # Soft Sage Fill
INK = colors.HexColor("#1A2421")           # Crisp Charcoal Body Text
MUTED = colors.HexColor("#5C6863")         # Subdued Slate for Labels/Footers
BORDER = colors.HexColor("#D2DCD7")        # Clean Light Divider
AMBER = colors.HexColor("#C27D20")         # Warm Ochre Warning Border
AMBER_BG = colors.HexColor("#FAF4E8")      # Soft Warm Callout Fill
CARD_BG = colors.HexColor("#F8FAF9")       # Light Card Background

DISCLAIMER_TEXT = (
    "For general informational purposes only — not a formal medical diagnosis and not a substitute for "
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
    return [("CLINICAL GUIDANCE", text)]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=PRIMARY,
            alignment=0,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "DocSub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=0,
        ),
        "h2": ParagraphStyle(
            "SectionH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=PRIMARY,
            spaceBefore=14,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=INK,
            spaceAfter=6,
        ),
        "callout": ParagraphStyle(
            "DocCallout",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12.5,
            textColor=INK,
        ),
        "muted": ParagraphStyle(
            "DocMuted",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
        "table_hdr": ParagraphStyle(
            "TableHdr",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=PRIMARY,
            alignment=1,
        ),
        "mono": ParagraphStyle(
            "DocMono",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "footer": ParagraphStyle(
            "DocFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=1,
        ),
    }


def _draw_header_footer(canvas, doc) -> None:
    """Draw running header (page 2+) and footer (all pages)."""
    canvas.saveState()
    page_w, page_h = A4
    margin = 16 * mm

    # Running Header on page 2+
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, page_h - 12 * mm, "DermOmni — Multimodal Clinical Consultation Report")
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(margin, page_h - 13.5 * mm, page_w - margin, page_h - 13.5 * mm)

    # Running Footer (all pages)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(margin, 15 * mm, page_w - margin, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(margin, 10 * mm, "DermOmni AI Skin Assistant — Educational & Informational Purpose Only")
    canvas.drawRightString(page_w - margin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# Maximum canvas bounds for images
_PHOTO_MAX_W = 178 * mm
_PHOTO_MAX_H = 95 * mm


def _find_embeddable_image(record: dict) -> Path | None:
    """Return the archived patient photo path when it exists and is readable."""
    candidate = (record or {}).get("media_image_path") or ""
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


def _scaled_photo_flowable(path: Path, max_w: float = _PHOTO_MAX_W, max_h: float = _PHOTO_MAX_H):
    """Build an aspect-preserved ReportLab Image flowable, or None on failure."""
    try:
        reader = ImageReader(str(path))
        iw, ih = reader.getSize()
        if not iw or not ih:
            return None
        scale = min(max_w / iw, max_h / ih, 1.0)
        return RLImage(str(path), width=iw * scale, height=ih * scale)
    except Exception:
        return None


def build_consultation_pdf(record: dict) -> bytes:
    """Render a stored consultation/research row to PDF bytes."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=f"DermOmni Report {record.get('id', '')[:8]}",
        author="DermOmni Health AI",
    )

    kind = record.get("kind", "consult")
    created = record.get("created_at", "")
    try:
        created_fmt = datetime.fromisoformat(str(created)).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        created_fmt = str(created or "—")

    story = []

    # Title & Subtitle Header
    story.append(Paragraph("DermOmni", styles["title"]))
    doc_type_name = "Evidence-Based Research Report" if kind == "research" else "Consultation Guidance Summary"
    story.append(Paragraph(
        f"{doc_type_name} &nbsp;•&nbsp; {created_fmt} &nbsp;•&nbsp; Ref ID: <font color=\"#0F4C3A\"><b>{str(record.get('id', ''))[:8]}</b></font>",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))

    # Meta Data Table (4 columns)
    evidence = record.get("evidence") or {}
    meta_rows = [
        [
            Paragraph("<b>Session Kind:</b>", styles["muted"]),
            Paragraph(kind.title(), styles["body"]),
            Paragraph("<b>Input Modality:</b>", styles["muted"]),
            Paragraph(str(record.get("media_kind", "none")).title(), styles["body"]),
        ],
        [
            Paragraph("<b>Evidence Grade:</b>", styles["muted"]),
            Paragraph(str(evidence.get("evidence_level", "—")).title(), styles["body"]),
            Paragraph("<b>Clinical Sources:</b>", styles["muted"]),
            Paragraph(str(evidence.get("source_count", len(record.get("sources") or []))), styles["body"]),
        ],
    ]
    meta_table = Table(meta_rows, colWidths=[30 * mm, 59 * mm, 30 * mm, 59 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # Disclaimer Callout Box
    disclaimer_table = Table([[
        Paragraph(f"<b>Clinical Disclaimer:</b> {DISCLAIMER_TEXT}", styles["callout"])
    ]], colWidths=[178 * mm])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, AMBER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 6))

    def _escape(text: str) -> str:
        """Escape plain text safely for ReportLab Paragraph markup."""
        escaped = text.replace("&", "&amp;")
        escaped = escaped.replace("<", "&lt;")
        escaped = escaped.replace(">", "&gt;")
        escaped = escaped.replace("\n", "<br/>")
        return escaped

    # Visual Comparison: Side-by-Side Table when both Original & Annotated images exist
    photo_path = _find_embeddable_image(record)
    annotated_path = _find_annotated_image(record)

    side_photo = _scaled_photo_flowable(photo_path, max_w=84 * mm, max_h=65 * mm) if photo_path else None
    side_annotated = _scaled_photo_flowable(annotated_path, max_w=84 * mm, max_h=65 * mm) if annotated_path else None

    if side_photo is not None and side_annotated is not None:
        story.append(Paragraph("Visual Analysis Comparison (Original vs Annotated Overlay)", styles["h2"]))
        comp_table = Table(
            [
                [
                    Paragraph("ORIGINAL PATIENT PHOTO", styles["table_hdr"]),
                    Paragraph("AI LESION OVERLAY & RISK HIGHLIGHTS", styles["table_hdr"]),
                ],
                [side_photo, side_annotated],
                [
                    Paragraph("Archived patient still photo", styles["callout"]),
                    Paragraph("<font color=\"#1F6F5C\">Green = Mild</font> &nbsp;|&nbsp; <font color=\"#C98A2C\">Amber = Moderate</font> &nbsp;|&nbsp; <font color=\"#C93B2B\">Red = Severe</font>", styles["callout"]),
                ],
            ],
            colWidths=[87 * mm, 87 * mm],
        )
        comp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_SOFT),
            ("BACKGROUND", (0, 1), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 6))
    elif photo_path is not None:
        full_photo = _scaled_photo_flowable(photo_path, max_w=_PHOTO_MAX_W, max_h=_PHOTO_MAX_H)
        if full_photo is not None:
            story.append(Paragraph("Uploaded Patient Photo", styles["h2"]))
            story.append(full_photo)
            story.append(Paragraph(
                "Photo provided by patient during consultation. Raw audio/video are never stored.",
                styles["muted"],
            ))
            story.append(Spacer(1, 4))
    elif annotated_path is not None:
        full_anno = _scaled_photo_flowable(annotated_path, max_w=_PHOTO_MAX_W, max_h=_PHOTO_MAX_H)
        if full_anno is not None:
            story.append(Paragraph("Annotated Analysis Overlay", styles["h2"]))
            story.append(full_anno)
            story.append(Paragraph(
                "AI-generated lesion callouts and severity regions (Green=Mild, Amber=Moderate, Red=Severe).",
                styles["muted"],
            ))
            story.append(Spacer(1, 4))

    # Patient Description / Speech Transcript
    patient_text = record.get("input_text") or record.get("transcript") or ""
    if patient_text:
        story.append(Paragraph("Patient Concern & Description", styles["h2"]))
        story.append(Paragraph(_escape(str(patient_text)[:3000]), styles["body"]))

    visual = (record.get("visual_analysis") or "").strip()
    main_text = (record.get("report") or record.get("guidance") or "").strip()

    if kind == "research":
        if visual:
            story.append(Paragraph("Multimodal Visual Findings", styles["h2"]))
            story.append(Paragraph(_escape(visual[:3000]), styles["body"]))
        for heading, body in parse_report_sections(main_text):
            story.append(Paragraph(heading.title(), styles["h2"]))
            story.append(Paragraph(_escape(body[:6000]), styles["body"]))
    else:
        if main_text:
            story.append(Paragraph("Clinical Guidance", styles["h2"]))
            story.append(Paragraph(_escape(main_text[:6000]), styles["body"]))
        if visual:
            story.append(Paragraph("Visual Findings", styles["h2"]))
            story.append(Paragraph(_escape(visual[:3000]), styles["body"]))

    # Sources Section
    sources = record.get("sources") or []
    if sources:
        story.append(Paragraph("Literature & Clinical Sources", styles["h2"]))
        items = []
        for src in sources[:20]:
            title = _escape(str(src.get("title", "Untitled"))[:200])
            url = _escape(str(src.get("url", ""))[:300])
            items.append(ListItem(
                [Paragraph(f"<b>{title}</b><br/><font color=\"#5C6863\">{url}</font>", styles["body"])],
                leftIndent=14,
                spaceBefore=2,
            ))
        story.append(ListFlowable(items, bulletType="1", start="1"))
    else:
        story.append(Paragraph("Literature & Clinical Sources", styles["h2"]))
        story.append(Paragraph("No external web sources were attached to this session.", styles["muted"]))

    # Evidence Assessment Box
    if evidence:
        story.append(Paragraph("Evidence & Grading Summary", styles["h2"]))
        ev_rows = [
            [
                Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", styles["muted"]),
                Paragraph(_escape(str(v)[:500]), styles["body"]),
            ]
            for k, v in evidence.items()
        ]
        ev_table = Table(ev_rows, colWidths=[48 * mm, 130 * mm])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))
    story.append(Paragraph(
        "Generated by DermOmni Health AI. "
        "Only patient still photos (when uploaded) are archived; "
        "raw voice and video recordings are never persisted.",
        styles["footer"],
    ))

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return buffer.getvalue()

