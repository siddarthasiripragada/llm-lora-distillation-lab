from sidsearch_lora_lab.protocol.rule_engine import apply_rules
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output
from sidsearch_lora_lab.distillation.pipeline import accepted_distilled_row
import json


def test_valid_rule_engine_output_passes_schema():
    output = apply_rules("Find pdf documents about AtlasSync owner Mira")
    result = validate_sidsearch_output(output)
    assert result.valid, result.errors


def test_invalid_intent_fails_schema():
    output = apply_rules("Find email about AtlasSync")
    output["intent"] = "bad"
    assert not validate_sidsearch_output(output).valid


def test_json_parser_rejects_non_json():
    parsed, error = parse_json_object("not json")
    assert parsed is None
    assert error and error.startswith("invalid_json")


def test_clarification_requires_question():
    output = apply_rules("Search latest")
    assert output["clarification_required"] is True
    assert output["clarification_question"]


def test_teacher_row_accepts_schema_valid_matching_output():
    expected = apply_rules("Find pptx documents about AtlasSync")
    row = {
        "example_id": "SS-test-0001",
        "dataset_version": "1.0.0",
        "protocol_version": "1.0.0",
        "category": "document_search",
        "difficulty": "easy",
        "input": "Find pptx documents about AtlasSync",
        "teacher_model": "qwen2.5:3b",
        "raw_output": json.dumps(expected),
    }
    accepted, rejected = accepted_distilled_row(row, __import__("pathlib").Path("data/protocol.md"))
    assert rejected is None
    assert accepted["expected_output"] == expected


def test_teacher_row_rejects_deterministic_mismatch():
    expected = apply_rules("Find pptx documents about AtlasSync")
    expected["source"] = "email"
    row = {
        "example_id": "SS-test-0002",
        "dataset_version": "1.0.0",
        "protocol_version": "1.0.0",
        "category": "document_search",
        "difficulty": "easy",
        "input": "Find pptx documents about AtlasSync",
        "teacher_model": "qwen2.5:3b",
        "raw_output": json.dumps(expected),
    }
    accepted, rejected = accepted_distilled_row(row, __import__("pathlib").Path("data/protocol.md"))
    assert accepted is None
    assert "deterministic_field_mismatch" in rejected["rejection_reasons"][0]
