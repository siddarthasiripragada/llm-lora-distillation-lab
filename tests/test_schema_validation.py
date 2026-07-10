from sidsearch_lora_lab.protocol.rule_engine import apply_rules
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output


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

