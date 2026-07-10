from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.generator import write_jsonl
from sidsearch_lora_lab.data.validator import read_jsonl, validate_dataset_rows
from sidsearch_lora_lab.distillation.pipeline import accepted_distilled_row


def main() -> None:
    raw_path = Path("data/distilled_raw.jsonl")
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print("no distilled_raw.jsonl yet; skipping")
        return
    accepted = []
    rejected = []
    for row in read_jsonl(raw_path):
        valid_row, rejected_row = accepted_distilled_row(row, Path("data/protocol.md"))
        if valid_row:
            accepted.append(valid_row)
        if rejected_row:
            rejected.append(rejected_row)
    write_jsonl(Path("data/distilled_accepted.jsonl"), accepted)
    write_jsonl(Path("data/rejected_examples.jsonl"), rejected)
    errors = validate_dataset_rows(accepted, Path("data/protocol.md"))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"accepted={len(accepted)} rejected={len(rejected)}")


if __name__ == "__main__":
    main()
