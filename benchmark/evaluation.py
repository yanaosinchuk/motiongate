"""Metrics and an offline replica of the scheduler.

Because scheduling policies only select among detector outputs, YOLO is run
once on every frame and each policy is replayed offline. The schedule function
mirrors MotionGatedDetector and is checked against it in unit tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

Box: TypeAlias = Sequence[float]
Boxes: TypeAlias = Sequence[Box]
FrameBoxes: TypeAlias = Sequence[Boxes]
SourceIndices: TypeAlias = Sequence[int] | np.ndarray


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(
    pred: Boxes,
    ref: Boxes,
    threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    """Greedy one-to-one matching by descending IoU."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    pairs = sorted(
        ((iou(p, r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_r: set[int] = set()
    ious: list[float] = []
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        ious.append(value)
    tp = len(ious)
    return tp, len(pred) - tp, len(ref) - tp, ious


def ratio(k: float, n: float) -> float:
    return k / n if n else math.nan


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Wilson interval requires 0 <= k <= n")
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def schedule(
    n: int,
    gate: Sequence[bool] | np.ndarray | None = None,
    refresh: int | None = None,
    min_interval: int = 1,
    fixed_rate: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a policy and return invocation flags and source-frame indices."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if min_interval < 1:
        raise ValueError("min_interval must be at least 1")
    if refresh is not None and refresh < 1:
        raise ValueError("refresh must be at least 1")
    if refresh is not None and min_interval > refresh:
        raise ValueError("min_interval must not exceed refresh")
    if fixed_rate is not None and fixed_rate < 1:
        raise ValueError("fixed_rate must be at least 1")
    if gate is not None and len(gate) < n:
        raise ValueError("gate sequence is shorter than n")

    invoked = np.zeros(n, dtype=bool)
    source = np.zeros(n, dtype=int)
    last, pending = -1, False
    for t in range(n):
        if fixed_rate is not None:
            z = t % fixed_rate == 0
        elif gate is None:
            z = True
        else:
            gap = t - last
            pending = pending or bool(gate[t])
            z = t == 0 or (pending and gap >= min_interval) or (refresh is not None and gap >= refresh)
        if z:
            last, pending = t, False
        invoked[t], source[t] = z, last
    return invoked, source


def evaluate_against_truth(
    truth: Sequence[Box | None],
    detections: FrameBoxes,
    source: SourceIndices,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate a controlled benchmark with at most one target box per frame."""
    tp = fn = fp = tn = box_fp = 0
    ious: list[float] = []
    hits: list[bool] = []
    empties: list[bool] = []
    for t, gt in enumerate(truth):
        shown = detections[int(source[t])]
        empties.append(len(shown) == 0)
        if gt is None:
            fp += int(bool(shown))
            tn += int(not shown)
            box_fp += len(shown)
            hits.append(False)
        else:
            m_tp, m_fp, m_fn, m_ious = match(shown, [gt], threshold)
            tp, fn, box_fp = tp + m_tp, fn + m_fn, box_fp + m_fp
            ious += m_ious
            hits.append(m_tp == 1)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "box_fp": box_fp,
        "ious": ious,
        "ages": [t - int(s) for t, s in enumerate(source)],
        "hits": hits,
        "empties": empties,
    }


def evaluate_against_reference(
    reference: FrameBoxes,
    source: SourceIndices,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compare retained boxes with the detector output on every frame."""
    tp = fp = fn = 0
    for t, ref in enumerate(reference):
        m_tp, m_fp, m_fn, _ = match(reference[int(source[t])], ref, threshold)
        tp, fp, fn = tp + m_tp, fp + m_fp, fn + m_fn
    recall, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision, "f1": f1}


def first_index(flags: Sequence[bool], start: int = 0) -> int | None:
    for t in range(start, len(flags)):
        if flags[t]:
            return t
    return None
