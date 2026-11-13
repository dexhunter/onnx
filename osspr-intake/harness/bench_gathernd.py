# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Baseline for onnx.reference.ops.op_gathernd._gather_nd_impl.

Workload: GatherND of 50,000 points from a (96, 64, 48) float32 volume.
Each index is a full coordinate, so every gathered value is a scalar.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from onnx.reference.ops.op_gathernd import _gather_nd_impl


def _workload():
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((96, 64, 48), dtype=np.float32)
    indices = np.stack(
        [
            rng.integers(0, 96, size=50_000),
            rng.integers(0, 64, size=50_000),
            rng.integers(0, 48, size=50_000),
        ],
        axis=1,
    ).astype(np.int64)
    return volume, indices


def main() -> None:
    volume, indices = _workload()

    def call():
        return _gather_nd_impl(volume, indices, 0)

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
    gathered = result[0]
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.2f}")
    print("samples_ms=" + ",".join(f"{sample:.3f}" for sample in ordered))
    print(f"checksum={float(gathered.reshape(-1)[0]):.8g}")


if __name__ == "__main__":
    main()
