from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from common.gemini import call_with_key_rotation, get_gemini_api_keys

load_dotenv()
logger = logging.getLogger(__name__)


def _generate_gemini_text(
    system_instruction: str,
    prompt: str,
    *,
    max_output_tokens: int = 800,
) -> str:
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")

    get_gemini_api_keys()  # fail fast with ConfigurationError when unconfigured
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    def _config() -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=max_output_tokens,
            # Keep the production-verified thinking configuration unchanged.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
            response_mime_type="text/plain",
        )

    def _op(client: genai.Client):
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=_config(),
        )

    response = call_with_key_rotation(
        _op,
        lambda key: genai.Client(api_key=key),
        operation_name="research-text",
    )

    if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
        usage = response.usage_metadata
        logger.warning(
            "Gemini response truncated by MAX_TOKENS (model=%s, thoughts_tokens=%s, output_tokens=%s, max_output_tokens=%s)",
            model,
            getattr(usage, "thoughts_token_count", None),
            getattr(usage, "candidates_token_count", None),
            max_output_tokens,
        )

    try:
        text = response.text
    except ValueError as exc:
        raise RuntimeError("Gemini returned no usable text.") from exc

    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def _clean_research_report(text: str) -> str:
    """Normalize Gemini output into plain text without markdown artifacts."""
    cleaned = text.replace("\u2019", "'").replace("```", "")
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^\s*(?:#+|[*+-]|\d+[.)])\s*", "", line)
        line = line.replace("**", "").replace("__", "")
        line = re.sub(r"[`*_#~]", "", line)
        line = re.sub(r"^Content:\s*", "", line, flags=re.IGNORECASE).strip()
        if line:
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return re.sub(r"\n\s+\n", "\n\n", cleaned)


class _SynthesisChain:
    def invoke(self, data: dict[str, str]) -> str:
        required = {"query", "visual_analysis", "research_data"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Missing synthesis fields: {sorted(missing)}")

        prompt = f"""Synthesise the data below into a structured skin research report.

Patient Query: {data['query']}

Visual Analysis (from image):
{data['visual_analysis']}

Medical Research:
{data['research_data']}

Write the report using exactly these four sections, each on its own line:

VISUAL OBSERVATIONS
Describe what was observed in the image in 2-3 clear sentences.

POTENTIAL CONDITIONS
Name the top 2-3 most likely conditions. For each, give a 2-sentence explanation covering what it is and why the visual evidence points to it.

RECOMMENDATIONS
Give 4 specific, actionable skin-care steps the patient can take at home, including ingredients to use or avoid and any lifestyle adjustments.

WHEN TO SEE A DOCTOR
Describe clearly the warning signs or symptoms that require an in-person dermatologist visit. Be specific — redness spreading beyond the border, pain, fever, or changes within 2 weeks are examples of when to escalate.

Write in full professional sentences throughout. Warm but clinical tone."""

        system_instruction = (
            "You are a clinical research dermatology assistant. "
            "Synthesise visual observations and research evidence into a clear, professional, "
            "educational skin condition report. Be specific and evidence-based. "
            "Never provide an official medical diagnosis. Output plain text only — no markdown, "
            "asterisks, or special characters, because this text may be converted to speech. "
            "The answer should be readable by a patient with no medical background."
        )
        return _clean_research_report(_generate_gemini_text(system_instruction, prompt, max_output_tokens=3000))


class _EvidenceChain:
    def invoke(self, data: dict[str, str]) -> str:
        required = {"report", "sources"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Missing evidence fields: {sorted(missing)}")

        prompt = f"""Evaluate the evidence quality of this skin research report.

Report:
{data['report']}

Sources used:
{data['sources']}

Respond in exactly this JSON format:
{{
  "evidence_level": "strong | moderate | limited",
  "evidence_reason": "One sentence explaining the rating.",
  "source_count": <integer — total unique sources>,
  "has_peer_reviewed": true | false,
  "confidence_note": "One sentence for the patient explaining what this evidence level means for them."
}}

Rating guide:
- strong   : 2 or more peer-reviewed PubMed / journal papers found
- moderate : Reputable medical sites found (AAD, Mayo Clinic, DermNet, NIH) but fewer than 2 papers
- limited  : Only general web sources, or fewer than 3 sources total"""

        system_instruction = (
            "You are a medical evidence quality evaluator. Assess the reliability and strength "
            "of research sources objectively. Respond ONLY with valid JSON — no preamble, no markdown fences."
        )
        return _generate_gemini_text(system_instruction, prompt, max_output_tokens=900)


synthesis_chain = _SynthesisChain()
evidence_chain = _EvidenceChain()
