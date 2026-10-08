"""Held-out real-video evaluation of the optional optical-flow extension.

The reference is YOLOv8m run on every frame of OpenCV's vtest.avi, matching the
paper's real-video protocol. Each candidate policy reuses those detector outputs
at the frames where its scheduler invokes the detector; skipped frames either
retain the previous boxes or propagate them with sparse optical flow.

This measures agreement with every-frame YOLO, not absolute detection accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
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
    UltralyticsPersonDetector,
    make_gate,
)

from .evaluation import match, ratio
from .settings import CONFIDENCE, IMAGE_SIZE, IOU_THRESHOLD, ROOT, VIDEO_M, VIDEO_REFRESH


class ReplayDetector:
    """Return cached every-frame detector output at the current frame index."""

    def __init__(self, reference: list[tuple[Detection, ...]]) -> None:
        self.reference = reference
        self.index = 0

    def __call__(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return self.reference[self.index]


def _encode(reference: list[tuple[Detection, ...]], ms: list[float], fps: float) -> dict[str, object]:
    return {
        "fps": fps,
        "ms": ms,
        "detections": [
            [
                {
                    "box": list(det.box),
                    "score": det.score,
                    "class_id": det.class_id,
                }
                for det in frame
            ]
            for frame in reference
        ],
    }


def _decode(data: dict[str, object]) -> tuple[list[tuple[Detection, ...]], list[float], float]:
    raw = data["detections"]
    reference = [
        tuple(
            Detection(tuple(float(v) for v in item["box"]), float(item["score"]), int(item["class_id"]))
            for item in frame
        )
        for frame in raw
    ]
    return reference, [float(v) for v in data["ms"]], float(data["fps"])


def every_frame_reference(args) -> tuple[list[tuple[Detection, ...]], list[float], float]:
    cache = Path(args.cache)
    if cache.exists() and not args.no_cache:
        data = json.loads(cache.read_text())
        reference, det_ms, fps = _decode(data)
        if reference and len(reference) == len(det_ms):
            print(f"Loaded {len(reference)} cached detector frames.", flush=True)
            return reference, det_ms, fps

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0

    detector = UltralyticsPersonDetector(
        str(args.weights),
        confidence=CONFIDENCE,
        image_size=IMAGE_SIZE,
        device=args.device,
    )
    detector.warmup()

    reference: list[tuple[Detection, ...]] = []
    det_ms: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        start = time.perf_counter()
        detections = tuple(detector(frame))
        det_ms.append(1000.0 * (time.perf_counter() - start))
        reference.append(detections)
        if len(reference) % 100 == 0:
            print(f"YOLO reference: {len(reference)} frames", flush=True)
        if args.max_frames and len(reference) >= args.max_frames:
            break
    capture.release()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(_encode(reference, det_ms, fps)))
    return reference, det_ms, fps


def evaluate_policy(
    args,
    reference: list[tuple[Detection, ...]],
    det_ms: list[float],
    *,
    min_interval: int,
    tracked: bool,
) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {args.video}")

    replay = ReplayDetector(reference)
    tracker = SparseOpticalFlowTracker() if tracked else None
    system = MotionGatedDetector(
        replay,
        make_gate("stabilized"),
        SchedulerConfig(VIDEO_REFRESH, min_interval),
        tracker=tracker,
    )

    tp = fp = fn = 0
    iou_sum = 0.0
    reference_boxes = 0
    invocations = 0
    tracking_refreshes = 0
    gate_ms = tracker_ms = 0.0
    selected_detector_ms = 0.0
    t = 0

    while t < len(reference):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        replay.index = t
        result = system.process(frame)

        pred_boxes = [det.box for det in result.detections]
        ref_boxes = [det.box for det in reference[t]]
        m_tp, m_fp, m_fn, matched_ious = match(pred_boxes, ref_boxes, IOU_THRESHOLD)
        tp += m_tp
        fp += m_fp
        fn += m_fn
        iou_sum += sum(matched_ious)
        reference_boxes += len(ref_boxes)

        if result.fresh:
            invocations += 1
            selected_detector_ms += det_ms[t]
        tracking_refreshes += int(result.reason == "tracking")
        gate_ms += result.gate_ms
        tracker_ms += result.tracker_ms
        t += 1

    capture.release()
    if t != len(reference):
        raise RuntimeError(f"video ended after {t} frames but reference contains {len(reference)}")

    recall = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    total_ms = selected_detector_ms + gate_ms + tracker_ms
    every_ms = float(np.sum(det_ms))
    return {
        "mode": "flow" if tracked else "retain",
        "M": min_interval,
        "K": VIDEO_REFRESH,
        "frames": t,
        "reference_boxes": reference_boxes,
        "invocations": invocations,
        "invocation_ratio": invocations / t,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_iou_matches": iou_sum / tp if tp else math.nan,
        "mean_iou_reference": iou_sum / reference_boxes if reference_boxes else math.nan,
        "tracking_refreshes": tracking_refreshes,
        "gate_ms_per_frame": gate_ms / t,
        "tracker_ms_per_frame": tracker_ms / t,
        "selected_detector_s": selected_detector_ms / 1000.0,
        "estimated_total_s": total_ms / 1000.0,
        "speedup": every_ms / total_ms if total_ms else math.inf,
    }


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    retain = df[df["mode"] == "retain"].set_index("M")
    flow = df[df["mode"] == "flow"].set_index("M")
    rows = []
    for m in sorted(set(retain.index) & set(flow.index)):
        rows.append(
            {
                "M": int(m),
                "delta_invocation_ratio": float(flow.loc[m, "invocation_ratio"] - retain.loc[m, "invocation_ratio"]),
                "delta_recall": float(flow.loc[m, "recall"] - retain.loc[m, "recall"]),
                "delta_precision": float(flow.loc[m, "precision"] - retain.loc[m, "precision"]),
                "delta_mean_iou_reference": float(
                    flow.loc[m, "mean_iou_reference"] - retain.loc[m, "mean_iou_reference"]
                ),
                "delta_speedup": float(flow.loc[m, "speedup"] - retain.loc[m, "speedup"]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    labels = {"retain": "retain last box", "flow": "optical flow"}

    for mode in ("retain", "flow"):
        sub = df[df["mode"] == mode].sort_values("invocation_ratio")
        axes[0].plot(sub["invocation_ratio"], sub["recall"], marker="o", label=labels[mode])
        axes[1].plot(
            sub["invocation_ratio"],
            sub["mean_iou_reference"],
            marker="o",
            label=labels[mode],
        )
        axes[2].plot(sub["invocation_ratio"], sub["speedup"], marker="o", label=labels[mode])

    axes[0].set(
        xlabel="detector invocation ratio",
        ylabel="recall vs every-frame YOLO",
        title="(a) Recall",
    )
    axes[1].set(
        xlabel="detector invocation ratio",
        ylabel="IoU sum / reference boxes",
        title="(b) Localisation + misses",
    )
    axes[2].set(
        xlabel="detector invocation ratio",
        ylabel="estimated speed-up",
        title="(c) Compute trade-off",
    )
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_tracking_video.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "vtest.avi")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov8m.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=ROOT / "results" / "cache" / "tracking_video_reference.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--m-values", type=int, nargs="+", default=list(VIDEO_M))
    args = parser.parse_args(argv)

    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if any(m < 1 or m > VIDEO_REFRESH for m in args.m_values):
        parser.error(f"each M must satisfy 1 <= M <= K={VIDEO_REFRESH}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, det_ms, fps = every_frame_reference(args)
    print(
        f"Reference: {len(reference)} frames, {fps:.1f} FPS, "
        f"mean detector time {np.mean(det_ms):.1f} ms/frame",
        flush=True,
    )

    rows = []
    for m in args.m_values:
        for tracked in (False, True):
            mode = "flow" if tracked else "retain"
            print(f"Running {mode}, K={VIDEO_REFRESH}, M={m}...", flush=True)
            rows.append(
                evaluate_policy(
                    args,
                    reference,
                    det_ms,
                    min_interval=m,
                    tracked=tracked,
                )
            )

    df = pd.DataFrame(rows)
    deltas = paired_deltas(df)
    df.to_csv(args.output_dir / "tracking_video.csv", index=False)
    deltas.to_csv(args.output_dir / "tracking_video_deltas.csv", index=False)
    make_figure(df, args.figure_dir)

    summary = {
        "method": (
            "Held-out OpenCV vtest.avi; reference is YOLOv8m on every frame. "
            "Metrics measure agreement with that reference, not absolute detection accuracy."
        ),
        "frames": len(reference),
        "fps": fps,
        "mean_detector_ms": float(np.mean(det_ms)),
        "gate": "stabilized",
        "refresh_K": VIDEO_REFRESH,
        "iou_threshold": IOU_THRESHOLD,
        "configurations": df.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
    }
    (args.output_dir / "tracking_video_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(df.to_string(index=False), flush=True)
    print("\nPaired deltas (flow - retain):", flush=True)
    print(deltas.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
