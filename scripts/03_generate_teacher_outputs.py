from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.data.generator import write_jsonl
from sidsearch_lora_lab.distillation.ollama_client import OllamaClient
from sidsearch_lora_lab.distillation.pipeline import build_teacher_prompt


def main() -> None:
    protocol_path = Path("data/protocol.md")
    seed_path = Path("data/seed_scenarios.jsonl")
    heldout_path = Path("benchmarks/heldout.jsonl")
    if not seed_path.exists():
        raise SystemExit("Run scripts/02_generate_seed_scenarios.py first.")
    protocol = protocol_path.read_text(encoding="utf-8")
    heldout_inputs = {row["input"] for row in read_jsonl(heldout_path)} if heldout_path.exists() else set()
    candidates = [row for row in read_jsonl(seed_path) if row["input"] not in heldout_inputs]
    client = OllamaClient()
    model = "qwen2.5:3b"
    raw_rows = []
    for index, row in enumerate(candidates[:175], start=1):
        prompt = build_teacher_prompt(protocol, row["input"])
        started = time.perf_counter()
        raw_output = client.generate(model=model, prompt=prompt, options={"temperature": 0.1, "seed": 42, "num_predict": 300})
        raw_rows.append({**row, "teacher_model": model, "raw_output": raw_output, "latency_seconds": time.perf_counter() - started})
        print(f"{index}/{min(175, len(candidates))} {row['example_id']}")
    write_jsonl(Path("data/distilled_raw.jsonl"), raw_rows)
    print(json.dumps({"wrote": len(raw_rows), "path": "data/distilled_raw.jsonl", "teacher_model": model}, indent=2))


if __name__ == "__main__":
    main()
