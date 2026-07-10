from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.generator import build_examples, write_jsonl


def main() -> None:
    rows = build_examples()
    write_jsonl(Path("data/seed_scenarios.jsonl"), rows)
    print(f"wrote {len(rows)} seed scenarios")


if __name__ == "__main__":
    main()
