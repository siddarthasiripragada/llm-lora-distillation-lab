from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.evaluation.metrics import aggregate_scores, composite_score, score_record
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output


Generator = Callable[[str], str]


SYSTEM_PROMPT = "You are a SidSearch planner. Return exactly one valid JSON object and no markdown."


def make_prompt(user_input: str, protocol_text: str | None = None) -> str:
    if protocol_text:
        return f"SYSTEM: {SYSTEM_PROMPT}\nSIDSEARCH PROTOCOL:\n{protocol_text}\nUSER: {user_input}\nASSISTANT:"
    return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {user_input}\nASSISTANT:"


def run_rule_engine_benchmark(benchmark_path: Path, output_path: Path, generator: Generator) -> dict[str, object]:
    rows = read_jsonl(benchmark_path)
    records = []
    scores = []
    for row in rows:
        started = time.perf_counter()
        raw_output = generator(row["input"])
        latency = time.perf_counter() - started
        parsed, parse_error = parse_json_object(raw_output)
        schema = validate_sidsearch_output(parsed or {}) if parsed else None
        score = score_record(row["expected_output"], parsed, json_valid=parse_error is None, schema_valid=bool(schema and schema.valid))
        scores.append(score)
        records.append({"example_id": row["example_id"], "raw_output": raw_output, "latency_seconds": latency, "scores": score})
    summary = aggregate_scores(scores)
    summary["composite_score"] = composite_score(summary)
    payload = {"summary": summary, "records": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_model_benchmark(
    benchmark_path: Path,
    output_path: Path,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: Path | None = None,
    protocol_path: Path | None = None,
    max_new_tokens: int = 256,
) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Model benchmark requires torch, transformers, and peft. "
            "Install requirements.txt in the active environment first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    protocol_text = protocol_path.read_text(encoding="utf-8") if protocol_path else None

    def generator(text: str) -> str:
        prompt = make_prompt(text, protocol_text)
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return run_rule_engine_benchmark(benchmark_path, output_path, generator)
