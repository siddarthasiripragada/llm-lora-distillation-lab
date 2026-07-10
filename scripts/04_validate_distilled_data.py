from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.validator import read_jsonl, validate_dataset_rows


def main() -> None:
    path = Path("data/distilled_accepted.jsonl")
    if not path.exists():
        print("no distilled_accepted.jsonl yet; skipping")
        return
    errors = validate_dataset_rows(read_jsonl(path), Path("data/protocol.md"))
    if errors:
        raise SystemExit("\n".join(errors))
    print("distilled data valid")


if __name__ == "__main__":
    main()
