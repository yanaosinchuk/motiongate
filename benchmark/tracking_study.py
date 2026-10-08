"""Controlled evaluation of the optional optical-flow extension.

This experiment deliberately uses an oracle detector: whenever the scheduler
requests a fresh detection, the exact synthetic ground-truth box is returned.
That isolates the effect of temporal box propagation from YOLO detector error.
The comparison is therefore between the original retain-last-box policy and the
same scheduler augmented with sparse optical flow.

Run:
    python -m benchmark.tracking_study
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from motiongate import (
    Detection,
    MotionGatedDetector,
    SchedulerConfig,
    SparseOpticalFlowTracker,
    TrackingConfig,
    make_gate,
)

from . import synthetic as syn
from .evaluation import iou
from .settings import IOU_THRESHOLD, REFRESH, ROOT

DEFAULT_M_VALUES = (1, 2, 4, 6, 8)


class OracleDetector:
    """Return the exact current synthetic target box when invoked."""

    def __init__(self) -> None:
        self.box: tuple[int, int, int, int] | None = None
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.box is None:
            return []
        return [Detection(tuple(float(v) for v in self.box), 1.0)]


@dataclass
class Accumulator:
    frames: int = 0
    positive: int = 0
    hits: int = 0
    stale_fp: int = 0
    invocations: int = 0
    tracking_refreshes: int = 0
    iou_sum_positive: float = 0.0
    iou_sum_hits: float = 0.0
    ages_sum: int = 0
    max_age: int = 0
    gate_ms: float = 0.0
    tracker_ms: float = 0.0

    def add(self, truth, result) -> None:
        self.frames += 1
        self.invocations += int(result.fresh)
        self.tracking_refreshes += int(result.reason == "tracking")
        self.ages_sum += result.detection_age
        self.max_age = max(self.max_age, result.detection_age)
        self.gate_ms += result.gate_ms
        self.tracker_ms += result.tracker_ms

        if truth is None:
            self.stale_fp += int(bool(result.detections))
            return

        self.positive += 1
        best = max((iou(det.box, truth) for det in result.detections), default=0.0)
        self.iou_sum_positive += best
        if best >= IOU_THRESHOLD:
            self.hits += 1
            self.iou_sum_hits += best

    def row(self, detector_ms: float) -> dict[str, float | int]:
        detector_cost = self.invocations * detector_ms
        total_cost = detector_cost + self.gate_ms + self.tracker_ms
        every_frame_cost = self.frames * detector_ms
        return {
            "frames": self.frames,
            "positive_frames": self.positive,
            "invocations": self.invocations,
            "invocation_ratio": self.invocations / self.frames,
            "recall": self.hits / self.positive if self.positive else math.nan,
            "mean_iou_positive": self.iou_sum_positive / self.positive if self.positive else math.nan,
            "mean_iou_hits": self.iou_sum_hits / self.hits if self.hits else math.nan,
            "stale_fp_frames": self.stale_fp,
            "mean_detection_age": self.ages_sum / self.frames,
            "max_detection_age": self.max_age,
            "tracking_refreshes": self.tracking_refreshes,
            "gate_ms_per_frame": self.gate_ms / self.frames,
            "tracker_ms_per_frame": self.tracker_ms / self.frames,
            "model_total_s": total_cost / 1000.0,
            "model_speedup": every_frame_cost / total_cost if total_cost else math.inf,
        }


def run_configuration(
    scene: syn.Scene,
    scenarios: list[syn.Scenario],
    *,
    seed: int,
    refresh: int,
    min_interval: int,
    tracked: bool,
    detector_ms: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    aggregate = Accumulator()
    scenario_rows: list[dict[str, float | int | str]] = []

    for scenario_index, scenario in enumerate(scenarios):
        detector = OracleDetector()
        tracker = SparseOpticalFlowTracker(TrackingConfig()) if tracked else None
        system = MotionGatedDetector(
            detector,
            make_gate("stabilized"),
            SchedulerConfig(refresh, min_interval),
            tracker=tracker,
        )
        local = Accumulator()

        for record in syn.render_scenario(
            scene,
            scenario,
            syn.scenario_rng(seed, scenario_index),
        ):
            detector.box = record.box
            result = system.process(record.frame)
            local.add(record.box, result)
            aggregate.add(record.box, result)

        local_row = local.row(detector_ms)
        local_row.update(
            {
                "mode": "flow" if tracked else "retain",
                "scenario": scenario.name,
                "M": min_interval,
                "K": refresh,
            }
        )
        scenario_rows.append(local_row)

    row = aggregate.row(detector_ms)
    row.update({"mode": "flow" if tracked else "retain", "M": min_interval, "K": refresh})
    return row, scenario_rows


def _paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["mode"] == "retain"].set_index("M")
    flow = df[df["mode"] == "flow"].set_index("M")
    rows = []
    for m in sorted(set(base.index) & set(flow.index)):
        rows.append(
            {
                "M": int(m),
                "delta_invocation_ratio": float(flow.loc[m, "invocation_ratio"] - base.loc[m, "invocation_ratio"]),
                "delta_recall": float(flow.loc[m, "recall"] - base.loc[m, "recall"]),
                "delta_mean_iou_positive": float(
                    flow.loc[m, "mean_iou_positive"] - base.loc[m, "mean_iou_positive"]
                ),
                "delta_speedup": float(flow.loc[m, "model_speedup"] - base.loc[m, "model_speedup"]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    labels = {"retain": "retain last box", "flow": "optical flow"}

    for mode in ("retain", "flow"):
        sub = df[df["mode"] == mode].sort_values("invocation_ratio")
        axes[0].plot(
            sub["invocation_ratio"],
            sub["recall"],
            marker="o",
            label=labels[mode],
        )
        axes[1].plot(
            sub["invocation_ratio"],
            sub["mean_iou_positive"],
            marker="o",
            label=labels[mode],
        )
        axes[2].plot(
            sub["invocation_ratio"],
            sub["model_speedup"],
            marker="o",
            label=labels[mode],
        )

    axes[0].set(
        xlabel="detector invocation ratio",
        ylabel=f"frame recall (IoU >= {IOU_THRESHOLD:g})",
        title="(a) Recall",
    )
    axes[1].set(
        xlabel="detector invocation ratio",
        ylabel="mean IoU on person frames",
        title="(b) Localisation",
    )
    axes[2].set(
        xlabel="detector invocation ratio",
        ylabel="modelled speed-up",
        title="(c) Compute trade-off",
    )
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_tracking_extension.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    df: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    detector_ms: float,
    seed: int,
) -> dict[str, object]:
    records = []
    for _, row in df.sort_values(["M", "mode"]).iterrows():
        records.append(
            {
                "mode": row["mode"],
                "M": int(row["M"]),
                "invocation_ratio": round(float(row["invocation_ratio"]), 6),
                "recall": round(float(row["recall"]), 6),
                "mean_iou_positive": round(float(row["mean_iou_positive"]), 6),
                "stale_fp_frames": int(row["stale_fp_frames"]),
                "tracker_ms_per_frame": round(float(row["tracker_ms_per_frame"]), 6),
                "model_speedup": round(float(row["model_speedup"]), 6),
            }
        )

    delta_records = [
        {
            "M": int(row["M"]),
            "delta_invocation_ratio": round(float(row["delta_invocation_ratio"]), 6),
            "delta_recall": round(float(row["delta_recall"]), 6),
            "delta_mean_iou_positive": round(float(row["delta_mean_iou_positive"]), 6),
            "delta_speedup": round(float(row["delta_speedup"]), 6),
        }
        for _, row in deltas.iterrows()
    ]

    return {
        "method": (
            "Controlled oracle-detector study: fresh detections use exact synthetic ground-truth boxes; "
            "differences therefore isolate temporal propagation and quality-triggered refresh."
        ),
        "gate": "stabilized",
        "refresh_K": REFRESH,
        "iou_threshold": IOU_THRESHOLD,
        "detector_cost_model_ms": detector_ms,
        "seed": seed,
        "configurations": records,
        "paired_deltas": delta_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seg-weights", default=str(ROOT / "weights" / "yolov8n-seg.pt"))
    parser.add_argument("--frames", type=int, default=syn.N_FRAMES)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--detector-ms",
        type=float,
        default=698.0,
        help="per-call detector cost used only for the speed-up model",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    args = parser.parse_args(argv)

    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.detector_ms <= 0:
        parser.error("--detector-ms must be positive")
    if any(m < 1 or m > REFRESH for m in args.m_values):
        parser.error(f"each M must satisfy 1 <= M <= K={REFRESH}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing controlled scene...", flush=True)
    scene = syn.prepare_scene(args.seg_weights)
    scenarios = syn.build_scenarios(args.frames)

    rows: list[dict[str, float | int | str]] = []
    scenario_rows: list[dict[str, float | int | str]] = []
    for m in args.m_values:
        for tracked in (False, True):
            mode = "flow" if tracked else "retain"
            print(f"Running {mode}, K={REFRESH}, M={m}...", flush=True)
            row, local = run_configuration(
                scene,
                scenarios,
                seed=args.seed,
                refresh=REFRESH,
                min_interval=m,
                tracked=tracked,
                detector_ms=args.detector_ms,
            )
            rows.append(row)
            scenario_rows.extend(local)

    df = pd.DataFrame(rows)
    scenarios_df = pd.DataFrame(scenario_rows)
    deltas = _paired_deltas(df)

    df.to_csv(args.output_dir / "tracking_controlled.csv", index=False)
    scenarios_df.to_csv(args.output_dir / "tracking_scenarios.csv", index=False)
    deltas.to_csv(args.output_dir / "tracking_deltas.csv", index=False)
    make_figure(df, args.figure_dir)

    summary = build_summary(df, deltas, detector_ms=args.detector_ms, seed=args.seed)
    (args.output_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(df.to_string(index=False), flush=True)
    print("\nPaired deltas (flow - retain):", flush=True)
    print(deltas.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
