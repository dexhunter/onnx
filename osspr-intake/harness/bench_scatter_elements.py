# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Baseline for onnx.reference.ops.op_scatter_elements.scatter_elements.

Workload: one ScatterElements update of a (32, 64, 64) float32 feature
volume along axis 1. Indices are one permutation of that axis, broadcast
across the batch and the last dimension, so every index along the scatter
axis is unique.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from onnx.reference.ops.op_scatter_elements import scatter_elements


def _workload():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((32, 64, 64), dtype=np.float32)
    updates = rng.standard_normal((32, 64, 64), dtype=np.float32)
    permutation = rng.permutation(64).astype(np.int64)
    indices = np.broadcast_to(permutation.reshape(1, 64, 1), data.shape).copy()
    return data, indices, updates


def main() -> None:
    data, indices, updates = _workload()

    def call():
        return scatter_elements(data, indices, updates, axis=1, reduction=None)

    call()
    gc.collect()
    samples_ms: list[float] = []
    result = None
    for _ in range(7):
        gc.collect()
        started = time.perf_counter()
        result = call()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples_ms)
    median = ordered[len(ordered) // 2]
    noise = (ordered[-1] - ordered[0]) / median * 100.0
    assert result is not None
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.2f}")
    print("samples_ms=" + ",".join(f"{sample:.3f}" for sample in ordered))
    print(f"checksum={float(result.reshape(-1)[0]):.8g}")


if __name__ == "__main__":
    main()
