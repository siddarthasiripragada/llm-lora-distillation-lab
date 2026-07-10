from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.config import write_json
from sidsearch_lora_lab.data.generator import build_examples, write_jsonl
from sidsearch_lora_lab.data.leakage import duplicate_inputs, suspicious_overlap
from sidsearch_lora_lab.data.splitter import category_counts, deterministic_split


def main() -> None:
    rows = build_examples()
    splits = deterministic_split(rows)
    write_jsonl(Path("data/train.jsonl"), splits["train"])
    write_jsonl(Path("data/validation.jsonl"), splits["validation"])
    write_jsonl(Path("benchmarks/heldout.jsonl"), splits["heldout"])
    write_jsonl(Path("data/distilled_accepted.jsonl"), [])
    write_jsonl(Path("data/distilled_raw.jsonl"), [])
    write_jsonl(Path("data/rejected_examples.jsonl"), [])
    findings = {
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "heldout": len(splits["heldout"]),
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
