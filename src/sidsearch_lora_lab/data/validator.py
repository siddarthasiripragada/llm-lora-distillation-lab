from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sidsearch_lora_lab.protocol.parser import known_rule_ids
from sidsearch_lora_lab.schemas import validate_sidsearch_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} root is not an object")
            rows.append(value)
    return rows


def validate_dataset_rows(rows: list[dict[str, Any]], protocol_path: Path) -> list[str]:
    errors: list[str] = []
    ids = set()
    rules = known_rule_ids(protocol_path)
    for row in rows:
        example_id = row.get("example_id")
        if example_id in ids:
            errors.append(f"duplicate_example_id:{example_id}")
        ids.add(example_id)
        expected = row.get("expected_output")
        if not isinstance(expected, dict):
            errors.append(f"{example_id}:missing_expected_output")
            continue
        result = validate_sidsearch_output(expected, rules)
        errors.extend(f"{example_id}:{error}" for error in result.errors)
    return errors

