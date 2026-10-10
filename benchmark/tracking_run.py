"""Run the post-paper retain-vs-optical-flow comparison.

This is intentionally separate from the paper-generation pipeline so the
published baseline remains reproducible and unchanged.

    python -m benchmark.tracking_run
    python -m benchmark.tracking_run --quick --fake-detector
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from motiongate import UltralyticsPersonDetector

from . import synthetic as syn
from .plotting import fig_tracking_comparison, style
from .run import controlled_study
from .runtime import BudgetExceeded, configure_budget, log
from .settings import CONFIDENCE, IMAGE_SIZE, ROOT
from .tracking import evaluate_tracking_modes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(ROOT / "weights" / "yolov8m.pt"))
    parser.add_argument("--seg-weights", default=str(ROOT / "weights" / "yolov8n-seg.pt"))
    parser.add_argument("--frames", type=int, default=syn.N_FRAMES)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fake-detector", action="store_true")
    parser.add_argument("--budget", type=float, default=0)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing detector/gate caches and recompute the comparison")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="PyTorch CPU threads for detector timing (default: 1)")
    args = parser.parse_args(argv)

    configure_budget(args.budget)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    if not args.fake_detector and args.device == "cpu":
        import torch
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if args.quick:
        args.frames, args.seeds = 90, 2

    for directory in ("results", "figures"):
        (ROOT / directory).mkdir(exist_ok=True)
    style()
    started = time.perf_counter()

    log("preparing controlled scene")
    scene = syn.prepare_scene(args.seg_weights)
    scenarios = syn.build_scenarios(args.frames)
    detector = None
    if not args.fake_detector:
        detector = UltralyticsPersonDetector(args.weights, CONFIDENCE, IMAGE_SIZE, args.device)
        detector.warmup()

    try:
        _, seed0 = controlled_study(args, scene, scenarios, detector)
    except BudgetExceeded as exc:
        log(f"time budget used up before: {exc}. Rerun the same command to resume from cache.")
        raise SystemExit(3)

    log("retain-vs-flow study")
    summary, per_scenario = evaluate_tracking_modes(args, scene, scenarios, seed0)
    summary.to_csv(ROOT / "results" / "tracking_controlled.csv", index=False)
    per_scenario.to_csv(ROOT / "results" / "tracking_scenarios.csv", index=False)

    import os
    import torch
    import ultralytics

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_min": round((time.perf_counter() - started) / 60, 2),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner": os.environ.get("RUNNER_NAME"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "weights": Path(args.weights).name,
        "seg_weights": Path(args.seg_weights).name,
        "fake_detector": args.fake_detector,
        "frames_per_scenario": args.frames,
        "scenarios": len(scenarios),
        "seed": args.seed,
        "comparison": ["retain", "flow"],
        "gate": "stabilized",
        "K": 15,
        "M": [1, 2, 4],
        "iou_threshold": 0.5,
    }
    (ROOT / "results" / "tracking_metadata.json").write_text(json.dumps(meta, indent=2))
    fig_tracking_comparison(summary)
    log(f"tracking comparison finished in {meta['runtime_min']} min")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
