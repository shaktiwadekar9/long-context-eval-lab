from __future__ import annotations

import re

from long_context_eval_lab.schemas import ModelResponse, Probe, Score


def normalize_text(text: str) -> str:
    """Normalize text for lightweight deterministic scoring."""

    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def contains_normalized(haystack: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(haystack)


def score_response(probe: Probe, response: ModelResponse) -> Score:
    """Score one model response.

    This v1 scorer is intentionally deterministic and simple. It is good for
    exact facts, IDs, names, regions, and NOT_FOUND checks. Later, need to add
    an LLM-as-judge scorer behind the same function signature.

    Args:
        probe: The evaluation probe with expected answer and scoring rules.
        response: The model's response to the probe.

    Returns:
        A Score object with the scoring results.
    """

    answer = response.answer or ""
    expected = probe.expected_answer

    forbidden_hit = any(contains_normalized(answer, wrong) for wrong in probe.forbidden_answers)

    if probe.answer_type == "not_found":
        answer_ok = normalize_text(answer) == "not_found" or "not_found" in normalize_text(answer)
    elif probe.answer_type == "exact":
        answer_ok = normalize_text(answer) == normalize_text(expected)
    else:
        answer_ok = contains_normalized(answer, expected)

    evidence_text = "\n".join(response.evidence)
    if probe.required_evidence:
        matched = sum(1 for item in probe.required_evidence if contains_normalized(evidence_text, item))
        evidence_score = matched / len(probe.required_evidence)
    else:
        evidence_score = 1.0 if answer_ok else 0.0

    # Hallucination signal: answer was expected to be missing, but model invented something.
    hallucination = probe.answer_type == "not_found" and not answer_ok

    passed = answer_ok and not forbidden_hit
    reason_parts: list[str] = []
    if not answer_ok:
        reason_parts.append("expected answer not found")
    if forbidden_hit:
        reason_parts.append("forbidden/stale answer used")
    if hallucination:
        reason_parts.append("model did not abstain")
    if not reason_parts:
        reason_parts.append("passed")

    return Score(
        passed=passed,
        answer_score=1.0 if answer_ok else 0.0,
        evidence_score=evidence_score,
        hallucination_flag=hallucination,
        forbidden_answer_hit=forbidden_hit,
        reason="; ".join(reason_parts),
    )
