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
import math
import pickle
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from motiongate import GateConfig, MotionGatedDetector, SchedulerConfig, UltralyticsPersonDetector, make_gate, theory  # noqa: E402

from . import synthetic as syn  # noqa: E402
from .evaluation import (evaluate_against_reference, evaluate_against_truth, first_index, ratio,  # noqa: E402
                         schedule, wilson)

from .plotting import fig_envelope, fig_pipeline, fig_scenarios, fig_tradeoff, style
from .policies import evaluate_policies, interpolate_fixed
from .settings import (
    CONFIDENCE,
    E2E_SCENARIOS,
    GATES,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    LABEL,
    REFRESH,
    ROOT,
    SHORT,
    SWEEP_DELTA,
    SWEEP_JITTER,
    SWEEP_SCALES,
    SWEEP_SIGMA,
    SWEEP_SPEED,
    VIDEO_FIXED_N,
    VIDEO_M,
    VIDEO_REFRESH,
)

BUDGET = {"seconds": None, "start": time.perf_counter()}


class BudgetExceeded(Exception):
    """Raised before an expensive stage once this invocation's time budget is used up."""


def check_budget(stage: str) -> None:
    if BUDGET["seconds"] and time.perf_counter() - BUDGET["start"] > BUDGET["seconds"]:
        raise BudgetExceeded(stage)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Controlled benchmark: gate study over seeds + YOLO pass on seed 0
# ---------------------------------------------------------------------------

def yolo_cache_key(args, scenario) -> str:
    import ultralytics
    raw = json.dumps([args.frames, args.seed, scenario.name, scenario.scale, scenario.jitter, IMAGE_SIZE,
                      CONFIDENCE, ultralytics.__version__, Path(args.weights).name, args.fake_detector])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def controlled_study(args, scene, scenarios, detector):
    import pickle
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
# 3. End-to-end validation of the online scheduler and of the cost model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 7. LaTeX export
# ---------------------------------------------------------------------------

def f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def export_latex(ctx):
    out = ROOT / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    macros = ["% Generated by benchmark/run.py -- do not edit by hand."]
    for name, value in ctx["macros"].items():
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (out / "numbers.tex").write_text("\n".join(macros) + "\n")
    for name, body in ctx["tables"].items():
        (out / f"{name}.tex").write_text(body)


def build_tables(gs, policies, per_scenario, scenarios, video):
    t = {}
    lines = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"Gate & Recall & FPR & Precision & Recall & FPR & FPR & Cost \\",
             r" & (all) & (all) & (all) & distant walker & illumination & jitter & (ms/frame) \\", r"\midrule"]
    for _, r in gs.iterrows():
        lines.append(f"{LABEL[r.gate]} & {f(r.recall)} & {f(r.fpr)} & {f(r.precision)} & {f(r.recall_small_walk)} & "
                     f"{f(r.fpr_illumination)} & {f(r.fpr_jitter)} & {f(r.gate_ms, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    t["tab_gates"] = "\n".join(lines) + "\n"

    def row(label, p):
        return (f"{label} & {f(p.ratio)} & {f(p.recall)} & {int(p.fp_frames)} & {f(p.mean_iou, 3)} & "
                f"{int(p.max_age)} & {f(p.entry_latency, 0)} & {f(p.exit_latency, 0)} & {f(p.est_s, 1)} & "
                f"{f(p.speedup, 2)} \\\\")
    lines = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
             r"Policy & $r$ & Recall & Stale & Mean & Max & Entry & Exit & Time & Speed- \\",
             r" & & & frames & IoU & age & lat. & lat. & (s) & up \\", r"\midrule"]
    lines.append(row("Every frame", policies[policies.family == "every"].iloc[0]))
    fds = policies[(policies.family == "gated") & (policies.gate == "stabilized") & (policies.K == REFRESH) & (policies.M == 1)].iloc[0]
    n_match = int(policies.loc[policies.family == "fixed"].iloc[(policies.loc[policies.family == "fixed", "ratio"] - fds.ratio).abs().argsort().iloc[0]].N)
    lines.append(row(f"Fixed rate, $N={n_match}$", policies[(policies.family == "fixed") & (policies.N == n_match)].iloc[0]))
    for g in GATES:
        p = policies[(policies.family == "gated") & (policies.gate == g) & (policies.K == REFRESH) & (policies.M == 1)].iloc[0]
        lines.append(row(f"Gated, {LABEL[g]}", p))
    lines += [r"\bottomrule", r"\end{tabular}"]
    t["tab_policies"] = "\n".join(lines) + "\n"

    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Invocation ratio $r$ ($K=" + str(REFRESH) + r"$)} & \multicolumn{3}{c}{Frame recall} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Scenario & FD & FD-S & MOG2 & FD & FD-S & MOG2 \\", r"\midrule"]
    for sc in scenarios:
        vals = [per_scenario[(per_scenario.policy == g) & (per_scenario.scenario == sc.name)].iloc[0] for g in GATES]
        lines.append(sc.label + " & " + " & ".join(f(v.ratio) for v in vals) + " & "
                     + " & ".join(f(v.recall) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    t["tab_scenarios"] = "\n".join(lines) + "\n"

    if video is not None:
        v = video[0]
        lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
                 r"Policy & $r$ & Recall & Precision & F1 & Speed-up \\", r"\midrule"]
        for _, p in v.iterrows():
            if p.family == "fixed" and p.N not in (1, 2, 3, 4):
                continue
            if p.family == "gated" and p.M not in (1, 2, 3):
                continue
            label = {"every": "Every frame", "fixed": f"Fixed rate, $N={int(p.N) if p.N == p.N else 0}$"}.get(
                p.family, f"Gated, {LABEL.get(p.gate, p.gate)}, $M={int(p.M)}$")
            if p.family == "fixed" and p.N == 1:
                continue
            lines.append(f"{label} & {f(p.ratio)} & {f(p.recall)} & {f(p.precision)} & {f(p.f1)} & {f(p.speedup, 2)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        t["tab_video"] = "\n".join(lines) + "\n"
    return t


def build_macros(args, gs, policies, e2e, agg, video, scene, cfg, seed0, pipeline_shift):
    m = {}
    m["NScen"], m["NFramesScen"] = str(len(seed0)), str(args.frames)
    m["NFramesCtrl"] = str(sum(len(v["truth"]) for v in seed0.values()))
    m["NSeeds"], m["RefreshK"] = str(args.seeds), str(REFRESH)
    m["NTransitions"] = str(int(gs.iloc[0][["tp", "fp", "fn", "tn"]].sum()))
    det_ms = np.concatenate([np.array(v["det_ms"]) for v in seed0.values()])
    m["YoloMs"] = f(float(np.mean(det_ms)), 0)
    for _, r in gs.iterrows():
        s = SHORT[r.gate]
        m[f"GateRecall{s}"], m[f"GateFpr{s}"], m[f"GatePrec{s}"] = f(r.recall), f(r.fpr), f(r.precision)
        m[f"GateRecallCI{s}"] = f"[{f(r.recall_lo)}, {f(r.recall_hi)}]"
        m[f"GateFprCI{s}"] = f"[{f(r.fpr_lo)}, {f(r.fpr_hi)}]"
        m[f"GateMs{s}"] = f(r.gate_ms, 2)
        m[f"CostRatio{s}"] = f(r.gate_ms / float(np.mean(det_ms)), 4)
        m[f"BreakEven{s}"] = f(theory.break_even_ratio(r.gate_ms, float(np.mean(det_ms))), 3)
        m[f"GateFprJitter{s}"], m[f"GateFprIllum{s}"] = f(r.fpr_jitter), f(r.fpr_illumination)
        m[f"GateRecallSmall{s}"], m[f"GateRecallJitterWalk{s}"] = f(r.recall_small_walk), f(r.recall_jitter_walk)
    for g in GATES:
        s = SHORT[g]
        p = policies[(policies.family == "gated") & (policies.gate == g) & (policies.K == REFRESH) & (policies.M == 1)].iloc[0]
        m[f"PolRatio{s}"], m[f"PolRecall{s}"], m[f"PolSpeed{s}"] = f(p.ratio), f(p.recall), f(p.speedup, 2)
        m[f"PolCalls{s}"], m[f"PolStale{s}"] = str(int(p.invocations)), str(int(p.fp_frames))
        m[f"PolExit{s}"], m[f"PolEntry{s}"] = f(p.exit_latency, 0), f(p.entry_latency, 0)
        m[f"FixedRecallAt{s}"] = f(interpolate_fixed(policies, p.ratio))
        m[f"PolTime{s}"] = f(p.est_s, 1)
    every = policies[policies.family == "every"].iloc[0]
    m["PolRecallEvery"], m["PolTimeEvery"], m["PolIouEvery"] = f(every.recall), f(every.est_s, 1), f(every.mean_iou)
    m["NegFrames"] = str(int(every.neg_frames))
    m["PosFrames"] = str(int(every.tp + every.fn))
    fds = policies[(policies.family == "gated") & (policies.gate == "stabilized") & (policies.K == REFRESH) & (policies.M == 1)].iloc[0]
    m["PolIouFDS"] = f(fds.mean_iou)
    tags = {2: "Two", 3: "Three", 4: "Four"}
    for g in GATES:
        s = SHORT[g]
        p1 = policies[(policies.family == "gated") & (policies.gate == g) & (policies.K == REFRESH) & (policies.M == 1)].iloc[0]
        m[f"PolIou{s}"] = f(p1.mean_iou)
        m[f"FixedIouAt{s}"] = f(interpolate_fixed(policies, p1.ratio, "mean_iou"))
        for mm in (2, 4):
            pm = policies[(policies.family == "gated") & (policies.gate == g) & (policies.K == REFRESH) & (policies.M == mm)].iloc[0]
            m[f"PolRatio{s}M{tags[mm]}"], m[f"PolRecall{s}M{tags[mm]}"] = f(pm.ratio), f(pm.recall)
            m[f"FixedRecallAt{s}M{tags[mm]}"] = f(interpolate_fixed(policies, pm.ratio))
    for n in (2, 3, 4):
        pf = policies[(policies.family == "fixed") & (policies.N == n)].iloc[0]
        m[f"PolIouFixed{tags[n]}"], m[f"PolRecallFixed{tags[n]}"] = f(pf.mean_iou), f(pf.recall)
        m[f"PolSpeedFixed{tags[n]}"] = f(pf.speedup, 2)
    if e2e is not None:
        for _, r in e2e.iterrows():
            s = SHORT[r.gate]
            m[f"EtoeMeasured{s}"], m[f"EtoeEstimated{s}"] = f(r.measured_s, 1), f(r.estimated_s, 1)
            m[f"EtoeErr{s}"] = f(100.0 * (r.measured_s - r.estimated_s) / r.estimated_s, 1)
            m[f"EtoeMismatch{s}"] = str(int(r.decision_mismatches))
            m[f"EtoeCalls{s}"], m["EtoeFrames"] = str(int(r.calls)), str(int(r.frames))
    for g, pipe in (("difference", "classic"), ("stabilized", "robust")):
        s = SHORT[g]
        m[f"PredSigmaLo{s}"] = f(theory.critical_sigma(cfg.threshold, 1e-2, pipe), 1)
        m[f"PredSigmaHi{s}"] = f(theory.critical_sigma(cfg.threshold, 1e-1, pipe), 1)
        m[f"PredSigmaPc{s}"] = f(theory.critical_sigma(cfg.threshold, theory.percolation_probability(3), pipe), 1)
        m[f"MeasSigma{s}"] = f(first_crossing(agg, "noise", g), 1)
        m[f"MeasIllum{s}"] = f(first_crossing(agg, "illumination", g), 1)
        m[f"MeasJitter{s}"] = f(first_crossing(agg, "jitter", g), 2)
        for scale in SWEEP_SCALES:
            tag = {0.15: "A", 0.25: "B", 0.7: "C"}[scale]
            m[f"MeasSpeed{s}{tag}"] = f(first_crossing(agg, "speed", g, scale, 0.5), 1)
    mean0, sd0 = theory.classic_noise_moments(1.0)
    m["PercProb"] = f(theory.percolation_probability(3), 4)
    m["BiasCoef"], m["KernelEnergy"], m["GrayEnergy"] = f(mean0, 3), f(theory.kernel_energy(5), 4), f(theory.gray_energy(), 4)
    m["RobustSdCoef"], m["ClassicSdCoef"] = f(theory.robust_noise_sd(1.0), 3), f(sd0, 3)
    for scale in SWEEP_SCALES:
        tag = {0.15: "A", 0.25: "B", 0.7: "C"}[scale]
        h = int(454 * scale)
        m[f"Height{tag}"] = str(h)
        m[f"PredSpeed{tag}"] = f(theory.area_model_critical_speed(h, cfg.min_area, cfg.dilation_iterations), 1)
    g_q = gradient_quantile(scene, cfg.min_area)
    m["GradQuantile"], m["PredJitter"] = f(g_q, 1), f(cfg.threshold / g_q, 2)
    m["PipelineShift"] = f(math.hypot(*pipeline_shift), 2)
    if video is not None:
        v, summ = video
        m["VtFrames"], m["VtFps"] = str(summ["frames"]), f(summ["fps"], 0)
        m["VtPersons"], m["VtEmpty"] = f(summ["mean_persons"], 1), str(summ["frames_without_person"])
        m["VtYoloMs"] = f(summ["det_ms"], 0)
        for g in GATES:
            s = SHORT[g]
            m[f"VtActive{s}"] = f(summ[f"active_{g}"])
            for mm in (1, 2, 3):
                p = v[(v.family == "gated") & (v.gate == g) & (v.M == mm)].iloc[0]
                tag = {1: "", 2: "Two", 3: "Three"}[mm]
                m[f"VtRatio{s}{tag}"], m[f"VtRecall{s}{tag}"] = f(p.ratio), f(p.recall)
                m[f"VtSpeed{s}{tag}"] = f(p.speedup, 2)
        for n in (2, 3, 4):
            p = v[(v.family == "fixed") & (v.N == n)].iloc[0]
            tag = {2: "Two", 3: "Three", 4: "Four"}[n]
            m[f"VtFixedRecall{tag}"], m[f"VtFixedSpeed{tag}"] = f(p.recall), f(p.speedup, 2)
    return m


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
    BUDGET["seconds"], BUDGET["start"] = (args.budget or None), time.perf_counter()
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
