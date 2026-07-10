from __future__ import annotations

from typing import Any


def f1(expected: list[str], actual: list[str]) -> float:
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set and not actual_set:
        return 1.0
    if not expected_set or not actual_set:
        return 0.0
    tp = len(expected_set & actual_set)
    precision = tp / len(actual_set)
    recall = tp / len(expected_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def filter_accuracy(expected: dict[str, str | None], actual: dict[str, str | None]) -> float:
    keys = ["end_date", "file_type", "owner", "start_date", "status"]
    return sum(1 for key in keys if expected.get(key) == actual.get(key)) / len(keys)


def score_record(expected: dict[str, Any], actual: dict[str, Any] | None, json_valid: bool = True, schema_valid: bool = True) -> dict[str, float]:
    if actual is None:
        return {
            "intent_accuracy": 0.0, "source_accuracy": 0.0, "confidence_accuracy": 0.0,
            "clarification_accuracy": 0.0, "applied_rule_f1": 0.0, "entity_f1": 0.0,
            "filter_accuracy": 0.0, "json_validity": float(json_valid), "schema_compliance": float(schema_valid),
            "exact_full_record_match": 0.0,
        }
    return {
        "intent_accuracy": float(expected["intent"] == actual.get("intent")),
        "source_accuracy": float(expected["source"] == actual.get("source")),
        "confidence_accuracy": float(expected["confidence"] == actual.get("confidence")),
        "clarification_accuracy": float(expected["clarification_required"] == actual.get("clarification_required")),
        "applied_rule_f1": f1(expected["applied_rules"], actual.get("applied_rules", [])),
        "entity_f1": f1(expected["entities"], actual.get("entities", [])),
        "filter_accuracy": filter_accuracy(expected["filters"], actual.get("filters", {})),
        "json_validity": float(json_valid),
        "schema_compliance": float(schema_valid),
        "exact_full_record_match": float(expected == actual),
    }


def aggregate_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def composite_score(scores: dict[str, float]) -> float:
    weights = {
        "intent_accuracy": 0.20,
        "source_accuracy": 0.10,
        "filter_accuracy": 0.20,
        "clarification_accuracy": 0.10,
        "applied_rule_f1": 0.15,
        "json_validity": 0.10,
        "schema_compliance": 0.10,
        "entity_f1": 0.05,
    }
    return sum(scores.get(key, 0.0) * weight for key, weight in weights.items())

