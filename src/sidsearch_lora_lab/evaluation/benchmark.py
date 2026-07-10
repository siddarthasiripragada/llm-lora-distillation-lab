from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.evaluation.metrics import aggregate_scores, composite_score, score_record
from sidsearch_lora_lab.schemas import parse_json_object, validate_sidsearch_output


Generator = Callable[[str], str]


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

