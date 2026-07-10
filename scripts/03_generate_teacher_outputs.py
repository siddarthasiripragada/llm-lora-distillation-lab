from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.distillation.pipeline import build_teacher_prompt


def main() -> None:
    protocol = Path("data/protocol.md").read_text(encoding="utf-8")
    rows = read_jsonl(Path("data/seed_scenarios.jsonl"))[:3]
    for row in rows:
        print(build_teacher_prompt(protocol, row["input"])[:500])
        print("---")
    print("Dry run only. Start Ollama and wire this script to OllamaClient for live teacher generation.")


if __name__ == "__main__":
    main()
