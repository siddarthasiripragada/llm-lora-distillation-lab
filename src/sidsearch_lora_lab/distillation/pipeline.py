from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sidsearch_lora_lab.protocol.parser import known_rule_ids
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

