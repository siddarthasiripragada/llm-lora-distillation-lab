from sidsearch_lora_lab.evaluation.bootstrap import bootstrap_ci
from sidsearch_lora_lab.evaluation.metrics import aggregate_scores, composite_score, f1, filter_accuracy, score_record


def test_f1_exact_match():
    assert f1(["a", "b"], ["a", "b"]) == 1.0


def test_f1_partial_match():
    assert round(f1(["a", "b"], ["b", "c"]), 3) == 0.5


def test_filter_accuracy_hand_calculated():
    expected = {"start_date": "2026-01-01", "end_date": None, "owner": "Sid", "file_type": None, "status": "open"}
    actual = {"start_date": "2026-01-01", "end_date": None, "owner": "Mira", "file_type": None, "status": "closed"}
    assert filter_accuracy(expected, actual) == 0.6


def test_score_record_exact_match():
    expected = {
        "intent": "email_search",
        "source": "email",
        "confidence": "high",
        "clarification_required": False,
        "applied_rules": ["SS-013"],
        "entities": ["AtlasSync"],
        "filters": {"start_date": None, "end_date": None, "owner": None, "file_type": None, "status": None},
    }
    scores = score_record(expected, expected)
    assert scores["exact_full_record_match"] == 1.0
    assert composite_score(scores) == 1.0


def test_aggregate_scores():
    assert aggregate_scores([{"a": 1.0}, {"a": 0.0}]) == {"a": 0.5}


def test_bootstrap_ci_returns_bounds():
    low, high = bootstrap_ci([0.0, 1.0, 1.0], samples=100)
    assert 0.0 <= low <= high <= 1.0

