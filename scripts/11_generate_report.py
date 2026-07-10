from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.reporting.report import generate_markdown_report


def main() -> None:
    generate_markdown_report(Path("results"), Path("results/report.md"))
    print("wrote results/report.md")


if __name__ == "__main__":
    main()
