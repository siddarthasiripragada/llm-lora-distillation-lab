import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.evaluation.benchmark import run_model_benchmark, run_rule_engine_benchmark
from sidsearch_lora_lab.protocol.rule_engine import apply_rules


def main() -> None:
    benchmark = Path("benchmarks/heldout.jsonl")
    try:
        closed = run_model_benchmark(benchmark, Path("results/base_closed_book_results.json"), protocol_path=None)
        open_book = run_model_benchmark(benchmark, Path("results/base_open_book_results.json"), protocol_path=Path("data/protocol.md"))
        print(json.dumps({"base_closed_book": closed["summary"], "base_open_book": open_book["summary"]}, indent=2, sort_keys=True))
        return
    except RuntimeError as exc:
        print(f"Model benchmark unavailable: {exc}")
        print("Running deterministic rule-engine pipeline validation instead.")

    def generator(text: str) -> str:
        return json.dumps(apply_rules(text), sort_keys=True)

    result = run_rule_engine_benchmark(benchmark, Path("results/base_rule_engine_results.json"), generator)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("This is a deterministic rule-engine baseline, not a model benchmark.")


if __name__ == "__main__":
    main()
