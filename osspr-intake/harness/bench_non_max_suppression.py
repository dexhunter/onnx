# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Baseline for NonMaxSuppression._run.

Workload: one image, 80 classes, 300 boxes. Boxes are corner-encoded
(center_point_box=0). Score threshold 0.5, IoU threshold 0.5, at most 20
boxes kept per class. This is the reference detector-head path.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from onnx import helper
from onnx.reference.ops.op_non_max_suppression import NonMaxSuppression


def _workload():
    rng = np.random.default_rng(0)
    n_boxes = 300
    boxes = np.zeros((1, n_boxes, 4), dtype=np.float32)
    y1 = rng.random(n_boxes).astype(np.float32)
    x1 = rng.random(n_boxes).astype(np.float32)
    boxes[0, :, 0] = y1
    boxes[0, :, 1] = x1
    boxes[0, :, 2] = y1 + 0.15
    boxes[0, :, 3] = x1 + 0.15
    scores = rng.random((1, 80, n_boxes), dtype=np.float32)
    node = helper.make_node(
        "NonMaxSuppression",
        [
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        ["selected"],
        center_point_box=0,
    )
    op = NonMaxSuppression(
        node,
        {"opsets": {"": 11}, "new_ops": None, "log": lambda *_args, **_kwargs: None},
    )
    return (
        op,
        boxes,
        scores,
        np.array([20], dtype=np.int64),
        np.array([0.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )


def main() -> None:
    op, boxes, scores, max_out, iou, score_thr = _workload()

    def call():
        return op._run(boxes, scores, max_out, iou, score_thr, 0)

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
    selected = result[0]
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.2f}")
    print("samples_ms=" + ",".join(f"{sample:.3f}" for sample in ordered))
    print(f"selected={int(selected.shape[0])}")


if __name__ == "__main__":
    main()
