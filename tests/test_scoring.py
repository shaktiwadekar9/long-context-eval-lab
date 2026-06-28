from long_context_eval_lab.core.scoring import score_response
from long_context_eval_lab.schemas import ModelResponse, Probe, ProbeCategory


def test_contains_answer_passes():
    """Test that a response containing the expected answer passes scoring."""
    probe = Probe(
        id="t1",
        category=ProbeCategory.NEEDLE_RETRIEVAL,
        question="db?",
        expected_answer="aurora-prod-17",
        answer_type="contains",
    )
    response = ModelResponse(answer="The database is aurora-prod-17.")
    score = score_response(probe, response)
    assert score.passed is True


def test_forbidden_answer_fails():
    """Test that a response containing a forbidden answer fails scoring."""
    probe = Probe(
        id="t2",
        category=ProbeCategory.DISTRACTOR_ROBUSTNESS,
        question="billing?",
        expected_answer="InvoiceCore",
        forbidden_answers=["BillFlow"],
        answer_type="contains",
    )
    response = ModelResponse(answer="Project Mercury uses BillFlow.")
    score = score_response(probe, response)
    assert score.passed is False
    assert score.forbidden_answer_hit is True


def test_not_found_passes():
    """Test that a response correctly indicating 'not found' passes scoring."""
    probe = Probe(
        id="t3",
        category=ProbeCategory.ABSTENTION,
        question="missing?",
        expected_answer="NOT_FOUND",
        answer_type="not_found",
    )
    response = ModelResponse(answer="NOT_FOUND")
    score = score_response(probe, response)
    assert score.passed is True
    assert score.hallucination_flag is False
