"""Run the complete evaluation and regenerate every table, figure and number of the paper.

    python -m benchmark.run                  # full run (about 25 min on one CPU core)
    python -m benchmark.run --quick --fake-detector   # 1-minute smoke test of the pipeline

Outputs: ``results/*.csv|json`` (raw numbers), ``figures/*.pdf|png`` and
``paper/generated/*.tex`` (LaTeX macros and tables read by the paper).  YOLO
outputs are cached in ``results/cache`` so that figures can be rebuilt quickly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from motiongate import GateConfig, UltralyticsPersonDetector, make_gate  # noqa: E402

from . import synthetic as syn  # noqa: E402
from .evaluation import ratio, wilson  # noqa: E402

from .experiments import end_to_end, envelope, video_study
from .plotting import fig_envelope, fig_pipeline, fig_scenarios, fig_tradeoff, style
from .policies import evaluate_policies
from .reporting import build_macros, build_tables, export_latex
from .runtime import BudgetExceeded, check_budget, configure_budget, log
from .settings import CONFIDENCE, GATES, IMAGE_SIZE, IOU_THRESHOLD, REFRESH, ROOT

# ---------------------------------------------------------------------------
# 1. Controlled benchmark: gate study over seeds + YOLO pass on seed 0
# ---------------------------------------------------------------------------

def yolo_cache_key(args, scenario) -> str:
    import ultralytics
    raw = json.dumps([args.frames, args.seed, scenario.name, scenario.scale, scenario.jitter, IMAGE_SIZE,
                      CONFIDENCE, ultralytics.__version__, Path(args.weights).name, args.fake_detector])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def controlled_study(args, scene, scenarios, detector):
    gate_rows, seed0 = [], {}
    seeds = list(range(args.seed, args.seed + args.seeds))
    cache_dir = ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps([yolo_cache_key(args, sc) for sc in scenarios] + seeds).encode()).hexdigest()[:12]
    study_cache = cache_dir / f"gate_study_{key}.pkl"
    if study_cache.exists():
        log("gate study loaded from cache")
        return pickle.loads(study_cache.read_bytes())
    for seed in seeds:
        for si, sc in enumerate(scenarios):
            first = seed == seeds[0]
            cache = cache_dir / f"yolo_{sc.name}.json"
            key = yolo_cache_key(args, sc)
            cached = None
            if first and cache.exists():
                data = json.loads(cache.read_text())
                cached = data if data.get("key") == key else None
            run_yolo = first and cached is None
            if run_yolo:
                check_budget(f"YOLO pass ({sc.name})")
            gates = {g: make_gate(g) for g in GATES}
            truth, motion, dets, det_ms = [], [], [], []
            active = {g: [] for g in GATES}
            gms = {g: [] for g in GATES}
            for rec in syn.render_scenario(scene, sc, syn.scenario_rng(seed, si)):
                truth.append(rec.box)
                motion.append(rec.motion)
                for g, gate in gates.items():
                    t0 = time.perf_counter()
                    res = gate.update(rec.frame)
                    gms[g].append(1000.0 * (time.perf_counter() - t0))
                    active[g].append(res.active)
                if run_yolo:
                    t0 = time.perf_counter()
                    if args.fake_detector:
                        out = [rec.box] if rec.box else []
                    else:
                        out = [d.box for d in detector(rec.frame)]
                    det_ms.append(1000.0 * (time.perf_counter() - t0))
                    dets.append([list(map(float, b)) for b in out])
            for g in GATES:
                pred, lab = np.array(active[g][1:]), np.array(motion[1:])
                gate_rows.append({"seed": seed, "scenario": sc.name, "gate": g,
                                  "tp": int(np.sum(pred & lab)), "fp": int(np.sum(pred & ~lab)),
                                  "fn": int(np.sum(~pred & lab)), "tn": int(np.sum(~pred & ~lab)),
                                  "gate_ms_mean": float(np.mean(gms[g][1:]))})
            if first:
                if run_yolo:
                    cache.write_text(json.dumps({"key": key, "boxes": dets, "ms": det_ms}))
                else:
                    dets, det_ms = cached["boxes"], cached["ms"]
                seed0[sc.name] = {"truth": truth, "motion": motion, "dets": dets, "det_ms": det_ms,
                                  "active": active, "gate_ms": gms}
            log(f"seed {seed} scenario {sc.name}: done" + (" (YOLO)" if run_yolo else ""))
    result = (pd.DataFrame(gate_rows), seed0)
    study_cache.write_bytes(pickle.dumps(result))
    return result


def summarise_gates(gate_df, scenarios):
    rows = []
    for g in GATES:
        sub = gate_df[gate_df.gate == g]
        tp, fp, fn, tn = (int(sub[c].sum()) for c in ("tp", "fp", "fn", "tn"))
        rec_lo, rec_hi = wilson(tp, tp + fn)
        fpr_lo, fpr_hi = wilson(fp, fp + tn)
        row = {"gate": g, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": ratio(tp, tp + fn),
               "recall_lo": rec_lo, "recall_hi": rec_hi, "precision": ratio(tp, tp + fp),
               "fpr": ratio(fp, fp + tn), "fpr_lo": fpr_lo, "fpr_hi": fpr_hi,
               "gate_ms": float(sub.gate_ms_mean.mean())}
        for sc in scenarios:
            s = sub[sub.scenario == sc.name]
            stp, sfp, sfn, stn = (int(s[c].sum()) for c in ("tp", "fp", "fn", "tn"))
            row[f"active_{sc.name}"] = ratio(stp + sfp, stp + sfp + sfn + stn)
            row[f"recall_{sc.name}"] = ratio(stp, stp + sfn)
            row[f"fpr_{sc.name}"] = ratio(sfp, sfp + stn)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov8m.pt"))
    ap.add_argument("--seg-weights", default=str(ROOT / "weights" / "yolov8n-seg.pt"))
    ap.add_argument("--video", default=str(ROOT / "data" / "vtest.avi"))
    ap.add_argument("--frames", type=int, default=syn.N_FRAMES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    ap.add_argument("--fake-detector", action="store_true", help="use ground-truth boxes instead of YOLO (testing only)")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--budget", type=float, default=0,
                    help="stop before expensive stages after this many seconds; rerun to resume from the cache")
    args = ap.parse_args(argv)
    configure_budget(args.budget)
    if args.quick:
        args.frames, args.seeds = 90, 2
    for d in ("results", "figures"):
        (ROOT / d).mkdir(exist_ok=True)
    style()
    cfg = GateConfig()
    t_start = time.perf_counter()

    log("preparing the controlled scene")
    scene = syn.prepare_scene(args.seg_weights)
    scenarios = syn.build_scenarios(args.frames)
    detector = None
    if not args.fake_detector:
        detector = UltralyticsPersonDetector(args.weights, CONFIDENCE, IMAGE_SIZE, args.device)
        detector.warmup()

    try:
        gate_df, seed0 = controlled_study(args, scene, scenarios, detector)
        e2e = None
        if not (args.skip_e2e or args.fake_detector or args.quick):
            e2e = end_to_end(args, scene, scenarios, detector, seed0)
            e2e.to_csv(ROOT / "results" / "end_to_end.csv", index=False)
        video = None
        if not args.skip_video and not args.fake_detector:
            log("real-video study")
            video = video_study(args, detector)
    except BudgetExceeded as exc:
        log(f"time budget used up before: {exc}. Progress is cached; rerun the same command to continue.")
        raise SystemExit(3)
    gate_df.to_csv(ROOT / "results" / "gate_runs.csv", index=False)
    gs = summarise_gates(gate_df, scenarios)
    gs.to_csv(ROOT / "results" / "gate_summary.csv", index=False)

    policies, per_scenario = evaluate_policies(seed0, scenarios)
    policies.to_csv(ROOT / "results" / "policies_controlled.csv", index=False)
    per_scenario.to_csv(ROOT / "results" / "scenarios_controlled.csv", index=False)


    log("operating-envelope sweeps")
    env_cache = ROOT / "results" / "cache" / "envelope.pkl"
    if env_cache.exists():
        agg = pickle.loads(env_cache.read_bytes())
    else:
        agg = envelope(scene, n=10 if args.quick else 30, seeds=(0,) if args.quick else (0, 1, 2))
        env_cache.write_bytes(pickle.dumps(agg))
    agg.to_csv(ROOT / "results" / "envelope.csv", index=False)

    if video is not None:
        video[0].to_csv(ROOT / "results" / "policies_video.csv", index=False)
        (ROOT / "results" / "video_summary.json").write_text(json.dumps(video[1], indent=2))

    log("figures and LaTeX export")
    fig_envelope(agg, scene, cfg)
    fig_tradeoff(policies, video)
    fig_scenarios(per_scenario, scenarios)
    shift = fig_pipeline(args, scene, scenarios, seed0)
    macros = build_macros(args, gs, policies, e2e, agg, video, scene, cfg, seed0, shift)
    export_latex({"macros": macros, "tables": build_tables(gs, policies, per_scenario, scenarios, video)})

    import torch
    import ultralytics
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_min": round((time.perf_counter() - t_start) / 60, 1),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(), "opencv": cv2.__version__, "numpy": np.__version__,
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "weights": Path(args.weights).name, "fake_detector": args.fake_detector,
            "source_person_box": list(scene.source_box),
            "parameters": {"frames_per_scenario": args.frames, "seeds": args.seeds, "seed": args.seed,
                           "noise_sigma": syn.NOISE_SIGMA, "threshold": cfg.threshold, "min_area": cfg.min_area,
                           "dilation": cfg.dilation_iterations, "blur_kernel": cfg.blur_kernel,
                           "refresh_K": REFRESH, "confidence": CONFIDENCE, "iou": IOU_THRESHOLD,
                           "image_size": IMAGE_SIZE}}
    (ROOT / "results" / "metadata.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "results" / "macros.json").write_text(json.dumps(macros, indent=2))
    log(f"finished in {meta['runtime_min']} min")


if __name__ == "__main__":
    main()
