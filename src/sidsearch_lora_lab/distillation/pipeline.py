from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sidsearch_lora_lab.protocol.parser import known_rule_ids
from sidsearch_lora_lab.protocol.rule_engine import apply_rules
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output


def build_teacher_prompt(protocol_text: str, scenario: str) -> str:
    return (
        "Use only the supplied SidSearch protocol. Return valid JSON only. "
        "Do not include markdown fences. Do not invent rule IDs.\n\n"
        f"SIDSEARCH PROTOCOL:\n{protocol_text}\n\nSCENARIO:\n{scenario}\n"
    )


def validate_teacher_row(raw_text: str, protocol_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    parsed, parse_error = parse_json_object(raw_text)
    if parse_error:
        return None, [parse_error]
    result = validate_sidsearch_output(parsed or {}, known_rule_ids(protocol_path))
    return (parsed if result.valid else None), result.errors


def accepted_distilled_row(raw_record: dict[str, Any], protocol_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parsed, errors = validate_teacher_row(str(raw_record.get("raw_output", "")), protocol_path)
    example_id = raw_record.get("example_id", "unknown")
    if parsed is None:
        return None, {**raw_record, "rejection_reasons": errors}
    expected_by_engine = apply_rules(str(raw_record.get("input", "")))
    deterministic_fields = ["intent", "source", "filters", "clarification_required"]
    mismatches = [field for field in deterministic_fields if parsed.get(field) != expected_by_engine.get(field)]
    if mismatches:
        return None, {
            **raw_record,
            "parsed_output": parsed,
            "rule_engine_expected": expected_by_engine,
            "rejection_reasons": [f"{example_id}:deterministic_field_mismatch:{','.join(mismatches)}"],
        }
    accepted = {
        "example_id": example_id,
        "dataset_version": raw_record["dataset_version"],
        "protocol_version": raw_record["protocol_version"],
        "category": raw_record["category"],
        "difficulty": raw_record["difficulty"],
        "input": raw_record["input"],
        "expected_output": parsed,
        "applied_rules": parsed["applied_rules"],
        "source": parsed["source"],
        "human_review_status": "teacher_generated_schema_validated",
        "teacher_model": raw_record.get("teacher_model"),
    }
    return accepted, None
