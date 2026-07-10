from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
PROTOCOL_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configuration files.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def git_commit_sha(root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return None if sha == "HEAD" else sha


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = ["torch", "transformers", "peft", "datasets", "trl", "accelerate", "pandas", "requests"]
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "git_commit": git_commit_sha(),
        "seed": SEED,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_version": DATASET_VERSION,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    benchmarks: Path = REPO_ROOT / "benchmarks"
    results: Path = REPO_ROOT / "results"
    adapters: Path = REPO_ROOT / "adapters"

