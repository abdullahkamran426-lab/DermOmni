from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

EvidenceLevel = Literal["strong", "moderate", "limited"]


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    url: HttpUrl


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evidence_level: EvidenceLevel
    evidence_reason: str = Field(min_length=1)
    source_count: int = Field(ge=0)
    has_peer_reviewed: bool
    confidence_note: str = Field(min_length=1)


class ResearchState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    visual_analysis: str
    visual_analysis_available: bool
    visual_analysis_error: str | None = None
    report: str = Field(min_length=1)
    report_generated_by: Literal["pending", "provider", "local_fallback"]
    report_error: str | None = None
    sources: list[Source] = Field(default_factory=list)
    research_papers: list[Source] = Field(default_factory=list)
    evidence: Evidence
    disclaimer: str = Field(min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("patient_query must not be empty")
        return text
