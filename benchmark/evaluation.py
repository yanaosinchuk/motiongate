"""Metrics and an offline replica of the scheduler.

Because every scheduling policy only *selects* among detector outputs, YOLO is
run once on every frame and each policy is replayed offline.  ``schedule``
mirrors ``MotionGatedDetector`` exactly (verified by a unit test).
"""

from __future__ import annotations

import math

import numpy as np


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
             + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def match(pred, ref, threshold: float = 0.5):
    """Greedy one-to-one matching by descending IoU -> (tp, fp, fn, matched IoUs)."""
    pairs = sorted(((iou(p, r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref)), reverse=True)
    used_p, used_r, ious = set(), set(), []
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
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def schedule(n: int, gate=None, refresh: int | None = None, min_interval: int = 1, fixed_rate: int | None = None):
    """Replay a policy.  Returns (invoked[bool], source[int]); source[t] = frame whose detections are shown."""
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


def evaluate_against_truth(truth, detections, source, threshold: float = 0.5) -> dict:
    """Controlled benchmark: one known target box per frame (or None)."""
    tp = fn = fp = tn = box_fp = 0
    ious, hits, empties = [], [], []
    for t, gt in enumerate(truth):
        shown = detections[source[t]]
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
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "box_fp": box_fp, "ious": ious,
            "ages": [t - int(s) for t, s in enumerate(source)], "hits": hits, "empties": empties}


def evaluate_against_reference(reference, source, threshold: float = 0.5) -> dict:
    """Real video: compare shown boxes with the every-frame detector output (the reference)."""
    tp = fp = fn = 0
    for t, ref in enumerate(reference):
        m_tp, m_fp, m_fn, _ = match(reference[source[t]], ref, threshold)
        tp, fp, fn = tp + m_tp, fp + m_fp, fn + m_fn
    recall, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision, "f1": f1}


def first_index(flags, start: int = 0):
    for t in range(start, len(flags)):
        if flags[t]:
            return t
    return None
