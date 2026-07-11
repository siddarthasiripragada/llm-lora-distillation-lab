from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.config import write_json
from sidsearch_lora_lab.data.generator import build_examples, write_jsonl
from sidsearch_lora_lab.data.leakage import duplicate_inputs, suspicious_overlap
from sidsearch_lora_lab.data.splitter import category_counts, deterministic_split
from sidsearch_lora_lab.data.validator import read_jsonl


def main() -> None:
    rows = build_examples()
    splits = deterministic_split(rows)
    accepted_path = Path("data/distilled_accepted.jsonl")
    heldout_path = Path("benchmarks/heldout.jsonl")
    accepted = read_jsonl(accepted_path) if accepted_path.exists() and accepted_path.stat().st_size > 0 else []
    if heldout_path.exists() and heldout_path.stat().st_size > 0:
        splits["heldout"] = read_jsonl(heldout_path)
    if len(accepted) >= 175:
        heldout_ids = {row["example_id"] for row in splits["heldout"]}
        heldout_inputs = {row["input"] for row in splits["heldout"]}
        usable_accepted = [
            row for row in accepted
            if row["example_id"] not in heldout_ids and row["input"] not in heldout_inputs
        ]
        if len(usable_accepted) < 175:
            raise SystemExit(
                f"not enough accepted non-benchmark rows: {len(usable_accepted)} available, 175 required"
            )
        splits["train"] = usable_accepted[:150]
        splits["validation"] = usable_accepted[150:175]
    write_jsonl(Path("data/train.jsonl"), splits["train"])
    write_jsonl(Path("data/validation.jsonl"), splits["validation"])
    write_jsonl(heldout_path, splits["heldout"])
    if not accepted_path.exists():
        write_jsonl(accepted_path, [])
    findings = {
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "heldout": len(splits["heldout"]),
        "training_source": "distilled_accepted" if len(accepted) >= 175 else "deterministic_rule_engine",
        "category_counts": {key: category_counts(value) for key, value in splits.items()},
        "duplicates": duplicate_inputs(rows),
        "suspicious_train_benchmark_overlap": suspicious_overlap(splits["train"], splits["heldout"], threshold=0.97),
    }
    write_json(Path("results/dataset_statistics.json"), findings)
    if findings["duplicates"] or findings["suspicious_train_benchmark_overlap"]:
        raise SystemExit(f"leakage check failed: {findings}")
    print(f"frozen benchmark: {findings}")


if __name__ == "__main__":
    main()
