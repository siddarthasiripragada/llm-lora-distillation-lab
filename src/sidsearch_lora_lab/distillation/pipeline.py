from __future__ import annotations

from pathlib import Path
from typing import Any

from sidsearch_lora_lab.protocol.parser import known_rule_ids
from sidsearch_lora_lab.protocol.rule_engine import apply_rules
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output


def build_teacher_prompt(protocol_text: str, scenario: str, deterministic_hint: dict[str, Any] | None = None) -> str:
    hint_text = ""
    if deterministic_hint:
        import json

        hint_text = (
            "\nREQUIRED DETERMINISTIC FIELDS:\n"
            "The following fields were computed by the SidSearch rule engine. "
            "You must preserve these exact values in your JSON answer:\n"
            f"{json.dumps(deterministic_hint, indent=2, sort_keys=True)}\n"
        )
    return (
        "You are generating one training label for the private SidSearch protocol.\n"
        "Use only the supplied SidSearch protocol and the scenario.\n"
        "Return valid JSON only. Do not include markdown fences or commentary.\n"
        "The JSON root must be one object with exactly these keys:\n"
        "intent, entities, source, filters, rewritten_query, confidence, "
        "clarification_required, clarification_question, applied_rules.\n\n"
        "Allowed intent values only:\n"
        "document_search, email_search, repository_search, combined_search, clarification_required.\n"
        "Allowed source values only: documents, email, github, all, unknown.\n"
        "Allowed confidence values only: high, medium, low.\n"
        "entities must be an array of strings, not objects.\n"
        "applied_rules must be an array of SS-### strings, not objects.\n"
        "filters must be an object with exactly these keys and string-or-null values:\n"
        "start_date, end_date, owner, file_type, status.\n"
        "Do not use intent=search. Do not use filters as an array. Do not invent rule IDs.\n\n"
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "intent": "document_search",\n'
        '  "entities": ["ExampleEntity"],\n'
        '  "source": "documents",\n'
        '  "filters": {\n'
        '    "start_date": null,\n'
        '    "end_date": null,\n'
        '    "owner": null,\n'
        '    "file_type": null,\n'
        '    "status": null\n'
        "  },\n"
        '  "rewritten_query": "ExampleEntity",\n'
        '  "confidence": "high",\n'
        '  "clarification_required": false,\n'
        '  "clarification_question": null,\n'
        '  "applied_rules": ["SS-004", "SS-015", "SS-019", "SS-020"]\n'
        "}\n\n"
        f"SIDSEARCH PROTOCOL:\n{protocol_text}\n\n"
        f"SCENARIO:\n{scenario}\n"
        f"{hint_text}\n"
        "JSON ONLY:"
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
