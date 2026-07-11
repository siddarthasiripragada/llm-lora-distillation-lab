from __future__ import annotations

import json
import random
from pathlib import Path

from sidsearch_lora_lab.evaluation.metrics import composite_score


def record_composite(record: dict[str, object]) -> float:
    return composite_score(record["scores"])  # type: ignore[arg-type]


def bootstrap_ci(values: list[float], seed: int = 42, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def load_records(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def main() -> None:
    base_records = load_records(Path("results/base_closed_book_results.json"))
    lora_records = load_records(Path("results/lora_closed_book_results.json"))

    if len(base_records) != len(lora_records):
        raise ValueError("Base and LoRA result files must contain the same number of examples.")

    base_scores = [record_composite(record) for record in base_records]
    lora_scores = [record_composite(record) for record in lora_records]
    paired_deltas = [lora - base for base, lora in zip(base_scores, lora_scores)]
    ci_low, ci_high = bootstrap_ci(paired_deltas)

    payload = {
        "n": len(paired_deltas),
        "base_closed_book_composite": sum(base_scores) / len(base_scores),
        "lora_closed_book_composite": sum(lora_scores) / len(lora_scores),
        "absolute_delta": sum(paired_deltas) / len(paired_deltas),
        "relative_delta": (sum(lora_scores) / sum(base_scores)) - 1,
        "paired_bootstrap_95_ci": [ci_low, ci_high],
        "samples": 10_000,
        "seed": 42,
    }

    output_path = Path("results/bootstrap_ci.json")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
