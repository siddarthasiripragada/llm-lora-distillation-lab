from __future__ import annotations

from typing import Any


def select_failures(records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: row.get("scores", {}).get("exact_full_record_match", 0.0))[:limit]


def select_improvements(base_records: list[dict[str, Any]], lora_records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in base_records}
    improved = []
    for row in lora_records:
        base = by_id.get(row["example_id"])
        if base and row.get("scores", {}).get("exact_full_record_match", 0) > base.get("scores", {}).get("exact_full_record_match", 0):
            improved.append(row)
    return improved[:limit]

