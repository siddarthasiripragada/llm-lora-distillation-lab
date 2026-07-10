from __future__ import annotations

import random


def bootstrap_ci(values: list[float], seed: int = 42, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]

