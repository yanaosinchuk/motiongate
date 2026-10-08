"""Benchmark experiment stages beyond the controlled gate study."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from motiongate import MotionGatedDetector, SchedulerConfig, make_gate

from . import synthetic as syn
from .evaluation import evaluate_against_reference, schedule, wilson
from .runtime import BudgetExceeded, check_budget, log
from .settings import (
    E2E_SCENARIOS,
    GATES,
    IOU_THRESHOLD,
    REFRESH,
    ROOT,
    SWEEP_DELTA,
    SWEEP_JITTER,
    SWEEP_SCALES,
    SWEEP_SIGMA,
    SWEEP_SPEED,
    VIDEO_FIXED_N,
    VIDEO_M,
    VIDEO_REFRESH,
)

def end_to_end(args, scene, scenarios, detector, seed0):
    """Time the online scheduler with the real detector on a scenario subset; compare with the cost model."""
    rows = []
    for g in ("difference", "stabilized"):
        cache = ROOT / "results" / "cache" / f"e2e_{g}.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        check_budget(f"end-to-end run ({g})")
        total_ms, est_ms, mismatches, calls, frames = 0.0, 0.0, 0, 0, 0
        for si, sc in enumerate(scenarios):
            if sc.name not in E2E_SCENARIOS:
                continue
            e = seed0[sc.name]
            invoked_offline, _ = schedule(len(e["truth"]), gate=e["active"][g], refresh=REFRESH)
            est_ms += float(np.sum(np.array(e["det_ms"])[invoked_offline])) + float(np.sum(e["gate_ms"][g]))
            system = MotionGatedDetector(detector, make_gate(g), SchedulerConfig(REFRESH, 1))
            for rec in syn.render_scenario(scene, sc, syn.scenario_rng(args.seed, si)):
                t0 = time.perf_counter()
                res = system.process(rec.frame)
                total_ms += 1000.0 * (time.perf_counter() - t0)
                mismatches += int(res.fresh != bool(invoked_offline[rec.index]))
                calls += int(res.fresh)
                frames += 1
        row = {"gate": g, "measured_s": total_ms / 1000.0, "estimated_s": est_ms / 1000.0,
               "calls": calls, "frames": frames, "decision_mismatches": mismatches}
        cache.write_text(json.dumps(row))
        rows.append(row)
        log(f"end-to-end {g}: {row['measured_s']:.1f} s measured, {row['estimated_s']:.1f} s estimated, "
            f"mismatches={mismatches}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Operating envelope (gate only)
# ---------------------------------------------------------------------------

def envelope(scene, n=30, seeds=(0, 1, 2)):
    rows = []
    specs = [("noise", v, None, lambda rng, v: syn.noise_sequence(scene, rng, v, n)) for v in SWEEP_SIGMA]
    specs += [("illumination", v, None, lambda rng, v: syn.illumination_sequence(scene, rng, v, n)) for v in SWEEP_DELTA]
    specs += [("jitter", v, None, lambda rng, v: syn.jitter_sequence(scene, rng, v, n)) for v in SWEEP_JITTER]
    specs += [("speed", v, s, (lambda s: lambda rng, v: syn.walk_sequence(scene, rng, v, s, n))(s))
              for s in SWEEP_SCALES for v in SWEEP_SPEED]
    for idx, (kind, value, scale, make) in enumerate(specs):
        for seed in seeds:
            frames = make(np.random.default_rng([seed, 7919, idx]), value)
            for g in ("difference", "stabilized"):
                gate = make_gate(g)
                acts = [gate.update(f).active for f in frames][1:]
                rows.append({"kind": kind, "value": value, "scale": scale, "gate": g, "seed": seed,
                             "active": int(np.sum(acts)), "n": len(acts)})
    df = pd.DataFrame(rows)
    agg = df.groupby(["kind", "value", "scale", "gate"], dropna=False)[["active", "n"]].sum().reset_index()
    agg["rate"] = agg.active / agg.n
    ci = [wilson(int(a), int(b)) for a, b in zip(agg.active, agg.n)]
    agg["lo"], agg["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return agg


def first_crossing(agg, kind, gate, scale=None, level=0.5):
    sub = agg[(agg.kind == kind) & (agg.gate == gate)]
    if scale is not None:
        sub = sub[np.isclose(sub.scale.astype(float), scale)]
    sub = sub.sort_values("value")
    values, rates = sub.value.to_numpy(float), sub.rate.to_numpy(float)
    above = np.nonzero(rates >= level)[0]
    if not len(above):
        return math.nan
    i = above[0]
    if i == 0 or rates[i] == rates[i - 1]:
        return float(values[i])
    # linear interpolation of the level crossing between two grid points
    return float(values[i - 1] + (level - rates[i - 1]) * (values[i] - values[i - 1]) / (rates[i] - rates[i - 1]))


def gradient_quantile(scene, min_area=900.0):
    gray = cv2.GaussianBlur(cv2.cvtColor(scene.background, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy).ravel()
    return float(np.quantile(mag, 1.0 - min_area / mag.size))


# ---------------------------------------------------------------------------
# 5. Real video (OpenCV's vtest.avi): agreement with the every-frame detector
# ---------------------------------------------------------------------------

def video_study(args, detector):
    path = Path(args.video)
    cache = ROOT / "results" / "cache" / "yolo_vtest.json"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        log(f"video {path} not found - skipping the real-video study")
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    cached = json.loads(cache.read_text()) if cache.exists() else {"boxes": [], "ms": []}
    gates = {g: make_gate(g) for g in GATES}
    active = {g: [] for g in GATES}
    gms = {g: [] for g in GATES}
    dets, det_ms, t = list(cached["boxes"]), list(cached["ms"]), 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        for g, gate in gates.items():
            t0 = time.perf_counter()
            active[g].append(gate.update(frame).active)
            gms[g].append(1000.0 * (time.perf_counter() - t0))
        if t >= len(dets):   # resume: detector outputs of earlier frames are cached
            try:
                check_budget("real-video YOLO pass")
            except BudgetExceeded:
                capture.release()
                cache.write_text(json.dumps({"boxes": dets, "ms": det_ms, "fps": fps}))
                raise
            t0 = time.perf_counter()
            dets.append([list(d.box) for d in detector(frame)] if detector else [])
            det_ms.append(1000.0 * (time.perf_counter() - t0))
        t += 1
        if args.quick and t >= 60:
            break
    capture.release()
    cache.write_text(json.dumps({"boxes": dets, "ms": det_ms, "fps": fps}))
    dets, det_ms = dets[:t], det_ms[:t]
    n = len(dets)
    rows = []
    pols = [("every", None, None, 1, None)] + [("fixed", None, None, 1, k) for k in VIDEO_FIXED_N]
    pols += [("gated", g, VIDEO_REFRESH, m, None) for g in GATES for m in VIDEO_M]
    base_ms = float(np.sum(det_ms))
    for family, gate, k, m, n_fixed in pols:
        if family == "every":
            invoked, source = schedule(n)
        elif family == "fixed":
            invoked, source = schedule(n, fixed_rate=n_fixed)
        else:
            invoked, source = schedule(n, gate=active[gate], refresh=k, min_interval=m)
        ev = evaluate_against_reference(dets, source, IOU_THRESHOLD)
        ms = float(np.sum(np.array(det_ms)[invoked])) + (float(np.sum(gms[gate])) if gate else 0.0)
        rows.append({"family": family, "gate": gate or "", "K": k, "M": m, "N": n_fixed,
                     "ratio": invoked.mean(), **ev, "speedup": base_ms / ms})
    summary = {"frames": n, "fps": fps, "mean_persons": float(np.mean([len(d) for d in dets])),
               "frames_without_person": int(sum(len(d) == 0 for d in dets)),
               "det_ms": float(np.mean(det_ms)),
               **{f"active_{g}": float(np.mean(active[g][1:])) for g in GATES},
               **{f"gate_ms_{g}": float(np.mean(gms[g][1:])) for g in GATES}}
    return pd.DataFrame(rows), summary


