"""Command-line interface for motion-gated person detection."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

from .config import GateConfig, SchedulerConfig, TrackingConfig
from .detector import UltralyticsPersonDetector
from .gates import GATE_NAMES, make_gate
from .render import draw_overlay
from .scheduler import MotionGatedDetector
from .tracker import TRACKER_NAMES, make_tracker


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motiongate", description=__doc__.splitlines()[0])
    p.add_argument("source", help="video file or integer camera index")
    p.add_argument("--weights", default="yolov8m.pt", help="Ultralytics checkpoint (default: yolov8m.pt)")
    p.add_argument("--gate", choices=GATE_NAMES, default="stabilized")
    p.add_argument("--threshold", type=float, default=20.0, help="difference threshold tau")
    p.add_argument("--min-area", type=float, default=900.0, help="minimum contour area A_min (pixels)")
    p.add_argument("--dilation", type=int, default=3, help="dilation iterations d")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--refresh", type=int, help="refresh interval K in frames")
    group.add_argument(
        "--max-staleness",
        type=float,
        default=1.0,
        help="maximum age of retained detections in seconds (default 1.0)",
    )
    p.add_argument("--min-interval", type=int, default=1, help="minimum frames between detector calls M")
    p.add_argument(
        "--tracker",
        choices=TRACKER_NAMES,
        default="none",
        help="box propagation between detector calls (default: none, reproduces the paper)",
    )
    p.add_argument("--tracking-min-quality", type=float, default=0.5)
    p.add_argument("--tracking-min-points", type=int, default=4)
    p.add_argument("--tracking-fb-threshold", type=float, default=1.5)
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, help="write an annotated video (mp4)")
    p.add_argument("--log", type=Path, help="write one CSV row per frame")
    p.add_argument("--show", action="store_true", help="display the annotated stream (Esc quits)")
    p.add_argument("--max-frames", type=_non_negative_int, default=0, help="stop after N frames (0 = no limit)")
    return p


def _scheduler_config(args: argparse.Namespace, fps: float) -> SchedulerConfig:
    if args.refresh is not None:
        return SchedulerConfig(args.refresh, args.min_interval)
    return SchedulerConfig.from_max_staleness(args.max_staleness, fps, args.min_interval)


def _ensure_parent(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: cannot open video source {args.source!r}", file=sys.stderr)
        return 2

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    try:
        scheduler_config = _scheduler_config(args, fps)
        gate_config = GateConfig(args.threshold, args.min_area, args.dilation)
        tracking_config = TrackingConfig(
            min_quality=args.tracking_min_quality,
            min_points=args.tracking_min_points,
            fb_threshold=args.tracking_fb_threshold,
        )
    except (TypeError, ValueError) as exc:
        capture.release()
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    gate = make_gate(args.gate, gate_config)
    try:
        detector = UltralyticsPersonDetector(args.weights, args.conf, args.imgsz, args.device)
    except (ImportError, ValueError) as exc:
        capture.release()
        print(f"error: cannot initialise detector: {exc}", file=sys.stderr)
        return 2
    tracker = make_tracker(args.tracker, tracking_config)
    system = MotionGatedDetector(detector, gate, scheduler_config, tracker=tracker)

    _ensure_parent(args.output)
    _ensure_parent(args.log)
    writer = None
    log_file = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(
            [
                "frame",
                "fresh",
                "reason",
                "detection_age",
                "tracked",
                "tracking_quality",
                "gate_active",
                "persons",
                "gate_ms",
                "tracker_ms",
                "detector_ms",
            ]
        )

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            result = system.process(frame)
            if log:
                log.writerow(
                    [
                        result.index,
                        int(result.fresh),
                        result.reason,
                        result.detection_age,
                        int(result.tracked),
                        "" if result.tracking_quality is None else f"{result.tracking_quality:.3f}",
                        int(result.gate.active),
                        len(result.detections),
                        f"{result.gate_ms:.3f}",
                        f"{result.tracker_ms:.3f}",
                        f"{result.detector_ms:.3f}",
                    ]
                )
            if args.output or args.show:
                canvas = draw_overlay(frame, result)
                if args.output:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                        if not writer.isOpened():
                            print(f"error: cannot open output video {args.output}", file=sys.stderr)
                            return 2
                    writer.write(canvas)
                if args.show:
                    cv2.imshow("motiongate", canvas)
                    if cv2.waitKey(1) == 27:
                        break
            if args.max_frames and result.index + 1 >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if log_file:
            log_file.close()
        if args.show:
            cv2.destroyAllWindows()

    s = system.stats
    if s.frames:
        print(
            f"frames={s.frames} detector_calls={s.invocations} ratio={s.invocation_ratio:.3f} "
            f"reasons={dict(s.reasons)} mean_ms/frame={s.total_ms / s.frames:.1f} "
            f"K={scheduler_config.refresh_interval} M={scheduler_config.min_interval}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
