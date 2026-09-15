"""Tests for the per-user vector-DB research memory + pipeline fallback."""

from __future__ import annotations

import pytest

from common.research_memory import (
    query_research_memory,
    reset_research_memory,
    save_research_memory,
)


@pytest.fixture()
def memory_env(tmp_path, monkeypatch):
    """Point the memory store at a tmp dir with the zero-dep JSONL backend."""
    monkeypatch.setenv("RESEARCH_MEMORY_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("RESEARCH_MEMORY_BACKEND", "jsonl")
    reset_research_memory()
    yield tmp_path
    reset_research_memory()


SAMPLE_REPORT = (
    "VISUAL OBSERVATIONS\nRed scaly plaques on extensor surfaces with silvery scales.\n\n"
    "POTENTIAL CONDITIONS\nPlaque psoriasis is a candidate given extensor distribution and scaling.\n\n"
    "RECOMMENDATIONS\nKeep the area moisturized with bland emollients and avoid scratching.\n\n"
    "WHEN TO SEE A DOCTOR\nSee a dermatologist if spreading, painful, or persistent beyond two weeks."
)


def test_save_and_query_roundtrip(memory_env):
    stored = save_research_memory(
        "user-1", "scaly elbow plaques", SAMPLE_REPORT,
        evidence_level="moderate", consultation_id="abc123",
        report_generated_by="provider",
    )
    assert stored > 0

    hits = query_research_memory("user-1", "elbow scaling plaques", top_k=3)
    assert len(hits) > 0
    assert "psoriasis" in hits[0]["content"].lower() or "plaques" in hits[0]["content"].lower()
    assert hits[0]["evidence_level"] == "moderate"


def test_user_isolation(memory_env):
    save_research_memory("user-1", "scaly plaques", SAMPLE_REPORT, report_generated_by="provider")
    assert query_research_memory("user-2", "scaly plaques") == []


def test_anonymous_never_stored_or_read(memory_env):
    assert save_research_memory("anonymous", "itchy rash", SAMPLE_REPORT, report_generated_by="provider") == 0
    save_research_memory("user-1", "itchy rash", SAMPLE_REPORT, report_generated_by="provider")
    assert query_research_memory("anonymous", "itchy rash") == []


def test_local_fallback_reports_are_not_ingested(memory_env):
    assert save_research_memory(
        "user-1", "itchy rash", SAMPLE_REPORT, report_generated_by="local_fallback"
    ) == 0
    assert query_research_memory("user-1", "itchy rash") == []


def _stub_chain(mocker, module_path, side_effect=None, return_value=None):
    stub = mocker.MagicMock()
    if side_effect is not None:
        stub.invoke.side_effect = side_effect
    else:
        stub.invoke.return_value = return_value
    mocker.patch(module_path, stub)
    return stub


def test_pipeline_backfills_memory_when_search_fails(memory_env, mocker):
    """Dead Tavily search + seeded memory -> synthesis LLM receives cached chunks."""
    import Skin_research_pipeline as pipeline

    save_research_memory(
        "user-9", "scaly elbow plaques", SAMPLE_REPORT,
        evidence_level="moderate", report_generated_by="provider",
    )
    mocker.patch.object(pipeline, "_run_searches", return_value=("", ""))
    captured = {}

    def _capture(payload):
        captured.update(payload)
        return "SYNTHESISED REPORT"

    _stub_chain(mocker, "Skin_research_pipeline.synthesis_chain", side_effect=_capture)
    _stub_chain(mocker, "Skin_research_pipeline.evidence_chain", return_value="{}")

    result = pipeline.run_skin_research_pipeline("scaly elbow plaques", user_id="user-9")

    assert result["report"] == "SYNTHESISED REPORT"
    assert "PRIOR RESEARCH (CACHED" in captured["research_data"]
    assert "psoriasis" in captured["research_data"].lower()


def test_pipeline_fallback_report_includes_memory(memory_env, mocker):
    """Dead search + dead synthesis -> local fallback still shows prior findings."""
    import Skin_research_pipeline as pipeline

    save_research_memory(
        "user-9", "scaly elbow plaques", SAMPLE_REPORT,
        evidence_level="moderate", report_generated_by="provider",
    )
    mocker.patch.object(pipeline, "_run_searches", return_value=("", ""))
    _stub_chain(mocker, "Skin_research_pipeline.synthesis_chain", side_effect=Exception("provider down"))
    _stub_chain(mocker, "Skin_research_pipeline.evidence_chain", return_value="{}")

    result = pipeline.run_skin_research_pipeline("scaly elbow plaques", user_id="user-9")

    assert result["report_generated_by"] == "local_fallback"
    assert "RELATED PRIOR FINDINGS" in result["report"]


def test_pipeline_without_user_id_skips_memory(memory_env, mocker):
    """No user -> no memory read; synthesis still works from live search text."""
    import Skin_research_pipeline as pipeline

    mocker.patch.object(pipeline, "_run_searches", return_value=("Title: Care\nURL: https://example.com", ""))
    captured = {}

    def _capture(payload):
        captured.update(payload)
        return "SYNTHESISED REPORT"

    _stub_chain(mocker, "Skin_research_pipeline.synthesis_chain", side_effect=_capture)
    _stub_chain(mocker, "Skin_research_pipeline.evidence_chain", return_value="{}")

    result = pipeline.run_skin_research_pipeline("itchy rash")

    assert result["report"] == "SYNTHESISED REPORT"
    assert "PRIOR RESEARCH (CACHED" in captured["research_data"]  # header present...
    assert "prior:" not in captured["research_data"]  # ...but no cached chunks
