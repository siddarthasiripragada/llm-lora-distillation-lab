from __future__ import annotations

import random
from collections import Counter
from typing import Any

from sidsearch_lora_lab.config import SEED


def deterministic_split(rows: list[dict[str, Any]], seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for category_index, (category, items) in enumerate(sorted(grouped.items())):
        shuffled = list(items)
        rng.shuffle(shuffled)
        heldout_count = 4 if category_index < 2 else 3
        heldout.extend(shuffled[:heldout_count])
        validation.extend(shuffled[heldout_count:heldout_count + 2])
        train.extend(shuffled[heldout_count + 2:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(heldout)
    validation_overflow = validation[25:]
    train.extend(validation_overflow)
    rng.shuffle(train)
    return {"train": train[:150], "validation": validation[:25], "heldout": heldout[:50]}


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["category"] for row in rows).items()))
