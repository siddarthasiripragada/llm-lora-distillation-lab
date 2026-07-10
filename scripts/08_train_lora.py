from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.training.train import run_cpu_smoke_test


def main() -> None:
    print("For local Windows, this runs the configured CPU smoke test. Use notebooks/train_lora_colab.ipynb for full training.")
    print(run_cpu_smoke_test())


if __name__ == "__main__":
    main()
