"""Controlled comparison of frozen retained boxes and optical-flow tracking.

The study re-renders the deterministic synthetic scenarios, reuses cached
per-frame detector outputs and timing measurements, and runs the online
scheduler in two modes:

- retain: the paper baseline, which reuses the last detector boxes;
- flow: the v1.1 sparse optical-flow tracker.

Detector outputs are therefore identical between modes. Any localisation
difference comes from how skipped frames are handled, while estimated runtime
includes the measured gate/tracker cost plus the cached detector cost at the
frames actually selected by the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from motiongate import Detection, MotionGatedDetector, SchedulerConfig, SparseOpticalFlowTracker, make_gate

from . import synthetic as syn
from .evaluation import match, ratio, wilson
from .settings import IOU_THRESHOLD, REFRESH


@dataclass
class ReplayDetector:
    """Return cached detector boxes for the frame selected by the benchmark."""

    boxes: list
    index: int = 0

    def __call__(self, frame):
        return tuple(Detection(tuple(map(float, box)), 1.0) for box in self.boxes[self.index])


def _evaluate_frame(gt, detections, threshold: float = IOU_THRESHOLD):
    shown = [d.box for d in detections]
    if gt is None:
        return {
            "tp": 0,
            "fn": 0,
            "fp_frame": int(bool(shown)),
            "tn": int(not shown),
            "box_fp": len(shown),
            "ious": [],
        }
    tp, box_fp, fn, ious = match(shown, [gt], threshold)
    return {"tp": tp, "fn": fn, "fp_frame": 0, "tn": 0, "box_fp": box_fp, "ious": ious}


def evaluate_tracking_modes(
    args,
    scene,
    scenarios,
    seed0,
    min_intervals=(1, 2, 4),
    gate_name="stabilized",
):
    """Run retain-vs-flow comparisons on the controlled benchmark."""

    rows = []
    scenario_rows = []
    for mode in ("retain", "flow"):
        for m in min_intervals:
            aggregate = {
                "tp": 0,
                "fn": 0,
                "fp_frames": 0,
                "tn": 0,
                "box_fp": 0,
                "frames": 0,
                "calls": 0,
                "gate_ms": 0.0,
                "tracker_ms": 0.0,
                "detector_ms": 0.0,
            }
            all_ious = []
            all_ages = []
            qualities = []
            tracked_frames = 0

            for si, sc in enumerate(scenarios):
                entry = seed0[sc.name]
                detector = ReplayDetector(entry["dets"])
                tracker = SparseOpticalFlowTracker() if mode == "flow" else None
                system = MotionGatedDetector(
                    detector,
                    make_gate(gate_name),
                    SchedulerConfig(REFRESH, m),
                    tracker=tracker,
                )

                local = {
                    "tp": 0,
                    "fn": 0,
                    "fp_frames": 0,
                    "tn": 0,
                    "box_fp": 0,
                    "calls": 0,
                    "gate_ms": 0.0,
                    "tracker_ms": 0.0,
                    "detector_ms": 0.0,
                }
                local_ious = []
                local_ages = []
                local_qualities = []
                local_tracked = 0

                records = syn.render_scenario(scene, sc, syn.scenario_rng(args.seed, si))
                for rec in records:
                    detector.index = rec.index
                    result = system.process(rec.frame)
                    ev = _evaluate_frame(rec.box, result.detections)
                    for key in ("tp", "fn", "fp_frames", "tn", "box_fp"):
                        source_key = "fp_frame" if key == "fp_frames" else key
                        local[key] += ev[source_key]
                    local_ious.extend(ev["ious"])
                    local_ages.append(result.detection_age)
                    local["calls"] += int(result.fresh)
                    local["gate_ms"] += result.gate_ms
                    local["tracker_ms"] += result.tracker_ms
                    if result.fresh:
                        local["detector_ms"] += float(entry["det_ms"][rec.index])
                    if result.tracked:
                        local_tracked += 1
                    if result.tracking_quality is not None:
                        local_qualities.append(float(result.tracking_quality))

                n_frames = len(entry["truth"])
                scenario_rows.append(
                    {
                        "mode": mode,
                        "M": m,
                        "scenario": sc.name,
                        "frames": n_frames,
                        "invocations": local["calls"],
                        "ratio": local["calls"] / n_frames,
                        "recall": ratio(local["tp"], local["tp"] + local["fn"]),
                        "mean_iou": float(np.mean(local_ious)) if local_ious else np.nan,
                        "fp_frames": local["fp_frames"],
                        "mean_age": float(np.mean(local_ages)),
                        "max_age": int(np.max(local_ages)),
                        "tracked_frames": local_tracked,
                        "tracking_quality": float(np.mean(local_qualities)) if local_qualities else np.nan,
                        "gate_ms": local["gate_ms"],
                        "tracker_ms": local["tracker_ms"],
                        "detector_ms": local["detector_ms"],
                        "est_s": (local["gate_ms"] + local["tracker_ms"] + local["detector_ms"]) / 1000.0,
                    }
                )

                for key in aggregate:
                    if key == "frames":
                        continue
                    aggregate[key] += local.get(key, 0)
                aggregate["frames"] += n_frames
                all_ious.extend(local_ious)
                all_ages.extend(local_ages)
                qualities.extend(local_qualities)
                tracked_frames += local_tracked

            rec_lo, rec_hi = wilson(aggregate["tp"], aggregate["tp"] + aggregate["fn"])
            total_ms = aggregate["gate_ms"] + aggregate["tracker_ms"] + aggregate["detector_ms"]
            rows.append(
                {
                    "mode": mode,
                    "gate": gate_name,
                    "K": REFRESH,
                    "M": m,
                    "frames": aggregate["frames"],
                    "invocations": aggregate["calls"],
                    "ratio": aggregate["calls"] / aggregate["frames"],
                    "recall": ratio(aggregate["tp"], aggregate["tp"] + aggregate["fn"]),
                    "recall_lo": rec_lo,
                    "recall_hi": rec_hi,
                    "box_precision": ratio(aggregate["tp"], aggregate["tp"] + aggregate["box_fp"]),
                    "mean_iou": float(np.mean(all_ious)) if all_ious else np.nan,
                    "fp_frames": aggregate["fp_frames"],
                    "mean_age": float(np.mean(all_ages)),
                    "max_age": int(np.max(all_ages)),
                    "tracked_frames": tracked_frames,
                    "tracking_quality": float(np.mean(qualities)) if qualities else np.nan,
                    "gate_ms": aggregate["gate_ms"],
                    "tracker_ms": aggregate["tracker_ms"],
                    "detector_ms": aggregate["detector_ms"],
                    "est_s": total_ms / 1000.0,
                }
            )

    summary = pd.DataFrame(rows)
    every_ms = float(sum(np.sum(np.asarray(entry["det_ms"], dtype=float)) for entry in seed0.values()))
    summary["speedup"] = every_ms / (summary["est_s"] * 1000.0)
    return summary, pd.DataFrame(scenario_rows)
