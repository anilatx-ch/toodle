"""Latency benchmark helpers."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile of a list of values."""
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def benchmark_model(
    predict_fn: Callable[[], object],
    n_warmup: int = 10,
    n_iter: int = 200,
) -> dict[str, float]:
    """Benchmark model inference latency."""
    for _ in range(n_warmup):
        predict_fn()

    timings_ms = []
    for _ in range(n_iter):
        start = time.perf_counter()
        predict_fn()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    total_sec = sum(timings_ms) / 1000.0
    throughput = n_iter / max(total_sec, 1e-9)

    return {
        "single_sample_p50_ms": _percentile(timings_ms, 50),
        "single_sample_p95_ms": _percentile(timings_ms, 95),
        "single_sample_mean_ms": float(statistics.mean(timings_ms)),
        "single_sample_std_ms": float(statistics.pstdev(timings_ms)),
        "throughput_samples_per_sec": float(throughput),
    }
