import json
from pathlib import Path


def main() -> None:
    results = {}
    for path in Path("results").glob("*_results.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        results[path.stem] = data.get("summary", {})
    output = Path("results/comparison.json")
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
