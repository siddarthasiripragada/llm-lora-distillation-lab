from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def duplicate_inputs(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        text = row["input"].strip().lower()
        if text in seen:
            duplicates.append(row["input"])
        seen.add(text)
    return duplicates


def suspicious_overlap(train: list[dict[str, Any]], benchmark: list[dict[str, Any]], threshold: float = 0.88) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for left in train:
        for right in benchmark:
            ratio = SequenceMatcher(None, left["input"].lower(), right["input"].lower()).ratio()
            if ratio >= threshold:
                findings.append({"train_id": left["example_id"], "benchmark_id": right["example_id"], "ratio": ratio})
    return findings

