from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


INTENTS = {"document_search", "email_search", "repository_search", "combined_search", "clarification_required"}
SOURCES = {"documents", "email", "github", "all", "unknown"}
CONFIDENCES = {"high", "medium", "low"}
FILTER_KEYS = {"start_date", "end_date", "owner", "file_type", "status"}
REQUIRED_OUTPUT_KEYS = {
    "intent",
    "entities",
    "source",
    "filters",
    "rewritten_query",
    "confidence",
    "clarification_required",
    "clarification_question",
    "applied_rules",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(value, dict):
        return None, "json_root_not_object"
    return value, None


def validate_sidsearch_output(value: dict[str, Any], known_rule_ids: set[str] | None = None) -> ValidationResult:
    errors: list[str] = []
    missing = REQUIRED_OUTPUT_KEYS - set(value)
    extra = set(value) - REQUIRED_OUTPUT_KEYS
    if missing:
        errors.append(f"missing_keys: {sorted(missing)}")
    if extra:
        errors.append(f"extra_keys: {sorted(extra)}")
    intent = value.get("intent")
    source = value.get("source")
    confidence = value.get("confidence")
    if not isinstance(intent, str) or intent not in INTENTS:
        errors.append("invalid_intent")
    if not isinstance(source, str) or source not in SOURCES:
        errors.append("invalid_source")
    if not isinstance(confidence, str) or confidence not in CONFIDENCES:
        errors.append("invalid_confidence")
    if not isinstance(value.get("entities"), list) or not all(isinstance(item, str) for item in value.get("entities", [])):
        errors.append("invalid_entities")
    filters = value.get("filters")
    if not isinstance(filters, dict):
        errors.append("invalid_filters")
    else:
        if set(filters) != FILTER_KEYS:
            errors.append("invalid_filter_keys")
        for key, item in filters.items():
            if key in FILTER_KEYS and item is not None and not isinstance(item, str):
                errors.append(f"invalid_filter_value:{key}")
    if not isinstance(value.get("rewritten_query"), str):
        errors.append("invalid_rewritten_query")
    if not isinstance(value.get("clarification_required"), bool):
        errors.append("invalid_clarification_required")
    question = value.get("clarification_question")
    if question is not None and not isinstance(question, str):
        errors.append("invalid_clarification_question")
    if value.get("clarification_required") and not question:
        errors.append("missing_clarification_question")
    if not value.get("clarification_required") and question is not None:
        errors.append("unexpected_clarification_question")
    applied_rules = value.get("applied_rules")
    if not isinstance(applied_rules, list) or not all(isinstance(item, str) for item in applied_rules or []):
        errors.append("invalid_applied_rules")
    elif known_rule_ids is not None:
        unknown = sorted(set(applied_rules) - known_rule_ids)
        if unknown:
            errors.append(f"unknown_rule_ids: {unknown}")
    return ValidationResult(valid=not errors, errors=errors)


def canonical_output(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": value["intent"],
        "entities": list(value["entities"]),
        "source": value["source"],
        "filters": {key: value["filters"].get(key) for key in sorted(FILTER_KEYS)},
        "rewritten_query": value["rewritten_query"],
        "confidence": value["confidence"],
        "clarification_required": value["clarification_required"],
        "clarification_question": value["clarification_question"],
        "applied_rules": list(value["applied_rules"]),
    }
