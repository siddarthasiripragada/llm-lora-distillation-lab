from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.evaluation.benchmark import run_model_benchmark


def main() -> None:
    adapter = Path("adapters/sidsearch-lora")
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit("LoRA benchmark requires a trained adapter at adapters/sidsearch-lora. Run Colab training first.")
    result = run_model_benchmark(
        Path("benchmarks/heldout.jsonl"),
        Path("results/lora_closed_book_results.json"),
        adapter_path=adapter,
        protocol_path=None,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
