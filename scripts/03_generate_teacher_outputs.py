from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.data.generator import write_jsonl
from sidsearch_lora_lab.distillation.ollama_client import OllamaClient
from sidsearch_lora_lab.distillation.pipeline import build_teacher_prompt
from sidsearch_lora_lab.protocol.rule_engine import apply_rules


TARGET_RAW_ROWS = 275


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
    raw_path = Path("data/distilled_raw.jsonl")
    raw_rows = read_jsonl(raw_path) if raw_path.exists() and raw_path.stat().st_size > 0 else []
    seen_ids = {row["example_id"] for row in raw_rows}
    target_raw_rows = min(TARGET_RAW_ROWS, len(candidates))
    remaining = [row for row in candidates if row["example_id"] not in seen_ids]
    if len(raw_rows) >= target_raw_rows:
        print(json.dumps({"already_have": len(raw_rows), "target": target_raw_rows, "path": str(raw_path)}, indent=2))
        return
    needed = target_raw_rows - len(raw_rows)
    for offset, row in enumerate(remaining[:needed], start=1):
        expected = apply_rules(row["input"])
        deterministic_hint = {
            "intent": expected["intent"],
            "source": expected["source"],
            "filters": expected["filters"],
            "clarification_required": expected["clarification_required"],
            "clarification_question": expected["clarification_question"],
        }
        prompt = build_teacher_prompt(protocol, row["input"], deterministic_hint=deterministic_hint)
        started = time.perf_counter()
        raw_output = client.generate(model=model, prompt=prompt, options={"temperature": 0.0, "seed": 42, "num_predict": 300})
        raw_rows.append({**row, "teacher_model": model, "raw_output": raw_output, "latency_seconds": time.perf_counter() - started})
        print(f"{len(raw_rows)}/{target_raw_rows} {row['example_id']} (+{offset}/{needed})")
        write_jsonl(raw_path, raw_rows)
    print(json.dumps({"wrote_total": len(raw_rows), "new_rows": min(needed, len(remaining)), "path": str(raw_path), "teacher_model": model}, indent=2))


if __name__ == "__main__":
    main()
