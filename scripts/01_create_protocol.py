from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidsearch_lora_lab.protocol.parser import parse_protocol


def main() -> None:
    rules = parse_protocol(Path("data/protocol.md"))
    print(f"protocol ok: {len(rules)} rules")


if __name__ == "__main__":
    main()
