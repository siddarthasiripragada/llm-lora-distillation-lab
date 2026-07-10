from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def write_placeholder_chart(path: Path, title: str, values: dict[str, float] | None) -> bool:
    if not values:
        LOGGER.warning("Skipping %s because no result data exists.", path.name)
        return False
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(values), list(values.values()))
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(values.values()) * 1.2))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True

