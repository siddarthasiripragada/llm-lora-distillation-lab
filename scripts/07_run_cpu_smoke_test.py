from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.training.train import run_cpu_smoke_test


def main() -> None:
    result = run_cpu_smoke_test()
    print(result)


if __name__ == "__main__":
    main()
