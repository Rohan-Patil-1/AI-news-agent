from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, HttpUrl


class TopStory(BaseModel):
    """A personalized story selected by the v0.3.0 ranking pipeline."""

    title: str = Field(min_length=1)
    link: HttpUrl
    description: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


class ResearchTask(BaseModel):
    """A focused investigation assigned by the research planner."""

    task_id: str = Field(min_length=1)
    story_title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    agent_type: Literal[
        "source_research",
        "technical_analysis",
        "source_comparison",
    ]
    priority: Literal["high", "medium", "low"] = "medium"
    search_queries: list[str] = Field(default_factory=list)
    required: bool = True


class SourceFinding(BaseModel):
    """A claim extracted from an external source."""

    finding_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    story_title: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: HttpUrl
    publisher: str = Field(min_length=1)
    published_at: datetime | None = None
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_type: Literal[
        "official",
        "news",
        "academic",
        "technical",
        "community",
        "other",
    ] = "other"
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class TechnicalAnalysis(BaseModel):
    """Technical details extracted when a story has technical relevance."""

    story_title: str = Field(min_length=1)
    is_technically_relevant: bool
    technologies: list[str] = Field(default_factory=list)
    architecture_details: list[str] = Field(default_factory=list)
    benchmark_details: list[str] = Field(default_factory=list)
    implementation_implications: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SourceComparison(BaseModel):
    """Comparison of claims and evidence across multiple sources."""

    story_title: str = Field(min_length=1)
    confirmed_claims: list[str] = Field(default_factory=list)
    disputed_claims: list[str] = Field(default_factory=list)
    unique_claims: list[str] = Field(default_factory=list)
    source_quality_notes: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class StorySynthesis(BaseModel):
    """A synthesized, evidence-aware conclusion for one story."""

    story_title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    technical_context: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    practical_implications: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchReport(BaseModel):
    """The final report generated for a research run."""

    report_id: str = Field(min_length=1)
    generated_at: datetime
    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    story_syntheses: list[StorySynthesis] = Field(default_factory=list)
    methodology: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)


class ResearchState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    run_id: str
    started_at: datetime
    top_stories: list[TopStory]
    research_tasks: list[ResearchTask]
    source_findings: list[SourceFinding]
    technical_analyses: list[TechnicalAnalysis]
    source_comparisons: list[SourceComparison]
    syntheses: list[StorySynthesis]
    report: ResearchReport
    errors: list[str]
    status: Literal[
        "pending",
        "planning",
        "researching",
        "analyzing",
        "synthesizing",
        "completed",
        "failed",
    ]