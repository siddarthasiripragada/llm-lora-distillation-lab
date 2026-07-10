from __future__ import annotations

import json
from pathlib import Path


def generate_markdown_report(results_dir: Path, output_path: Path) -> str:
    sections = ["# SidSearch LoRA Experiment Report", "", "This report is generated only from result files present on disk.", ""]
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "gradient_verification.json" or path.name.endswith("_results.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            sections.extend([f"## {path.name}", "```json", json.dumps(data.get("summary", data), indent=2, sort_keys=True), "```", ""])
    if len(sections) <= 4:
        sections.append("No benchmark result files were found. Do not claim benchmark improvement yet.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sections) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text

