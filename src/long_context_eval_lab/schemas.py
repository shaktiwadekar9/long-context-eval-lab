from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProbeCategory(str, Enum):
    """Supported long-context capability probes.

    Add new categories here when you want to expand the benchmark.
    The evaluator does not hard-code category behavior; categories are used
    for grouping, reporting, and analysis.
    """

    NEEDLE_RETRIEVAL = "needle_retrieval"
    LOST_IN_MIDDLE = "lost_in_middle"
    MULTI_HOP_QA = "multi_hop_qa"
    DISTRACTOR_ROBUSTNESS = "distractor_robustness"
    EFFECTIVE_CONTEXT_LENGTH = "effective_context_length"
    ABSTENTION = "abstention"


class Probe(BaseModel):
    """One evaluation question against a provided long context."""

    id: str
    category: ProbeCategory
    question: str
    expected_answer: str
    required_evidence: list[str] = Field(default_factory=list)
    forbidden_answers: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    answer_type: Literal["exact", "contains", "not_found"] = "contains"
    notes: str | None = None


class ModelResponse(BaseModel):
    """Normalized model response returned by every model client."""

    answer: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    raw_text: str = ""
    latency_seconds: float = 0.0
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    """Per-probe scoring result."""

    passed: bool
    answer_score: float
    evidence_score: float
    hallucination_flag: bool
    forbidden_answer_hit: bool
    reason: str


class EvalResult(BaseModel):
    """Stored result for one probe run."""

    run_id: str
    model: str
    provider: str
    context_file: str
    context_chars: int
    approx_input_tokens: int
    probe: Probe
    response: ModelResponse
    score: Score
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunSummary(BaseModel):
    """Aggregated report for a full evaluation run."""

    run_id: str
    model: str
    provider: str
    total: int
    passed: int
    accuracy: float
    hallucination_rate: float
    avg_latency_seconds: float
    by_category: dict[str, dict[str, float | int]]
