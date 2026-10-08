"""Offline scheduling-policy replay and evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .evaluation import evaluate_against_truth, first_index, ratio, schedule, wilson
from .settings import FIXED_N, GATES, IOU_THRESHOLD, K_VALUES, M_VALUES, REFRESH

def policy_list():
    pols = [("every", None, None, 1, None)]
    pols += [("fixed", None, None, 1, n) for n in FIXED_N]
    for g in GATES:
        pols += [("gated", g, k, 1, None) for k in K_VALUES]
        pols += [("gated", g, REFRESH, m, None) for m in M_VALUES if m != 1]
    return pols


def replay(entry, family, gate, k, m, n_fixed):
    n = len(entry["truth"])
    if family == "every":
        return schedule(n)
    if family == "fixed":
        return schedule(n, fixed_rate=n_fixed)
    return schedule(n, gate=entry["active"][gate], refresh=k, min_interval=m)


def evaluate_policies(seed0, scenarios):
    rows, per_scenario = [], []
    ese = "enter_stop_exit"
    ref_eval = evaluate_against_truth(seed0[ese]["truth"], seed0[ese]["dets"], np.arange(len(seed0[ese]["truth"])))
    ref_entry = first_index(ref_eval["hits"], 10)
    ref_exit = first_index(ref_eval["empties"], 60)
    for family, gate, k, m, n_fixed in policy_list():
        agg = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "box_fp": 0, "inv": 0, "frames": 0, "ms": 0.0}
        ious, ages = [], []
        entry_lat = exit_lat = math.nan
        for sc in scenarios:
            e = seed0[sc.name]
            invoked, source = replay(e, family, gate, k, m, n_fixed)
            ev = evaluate_against_truth(e["truth"], e["dets"], source, IOU_THRESHOLD)
            for c in ("tp", "fn", "fp", "tn", "box_fp"):
                agg[c] += ev[c]
            ious += ev["ious"]
            ages += ev["ages"]
            ms = float(np.sum(np.array(e["det_ms"])[invoked]))
            if family == "gated":
                ms += float(np.sum(e["gate_ms"][gate]))
            agg["ms"] += ms
            agg["inv"] += int(invoked.sum())
            agg["frames"] += len(invoked)
            if sc.name == ese:
                pe = first_index(ev["hits"], 10)
                entry_lat = (pe - ref_entry) if (pe is not None and ref_entry is not None) else math.nan
                px = first_index(ev["empties"], ref_exit) if ref_exit is not None else None
                exit_lat = (px - ref_exit) if px is not None else (len(source) - ref_exit if ref_exit else math.nan)
            if (family == "every") or (family == "gated" and k == REFRESH and m == 1):
                per_scenario.append({"policy": gate or family, "scenario": sc.name,
                                     "ratio": invoked.mean(), "recall": ratio(ev["tp"], ev["tp"] + ev["fn"]),
                                     "stale_fp": ev["fp"]})
        lo, hi = wilson(agg["tp"], agg["tp"] + agg["fn"])
        rows.append({"family": family, "gate": gate or "", "K": k, "M": m, "N": n_fixed,
                     "invocations": agg["inv"], "frames": agg["frames"], "ratio": agg["inv"] / agg["frames"],
                     "tp": agg["tp"], "fn": agg["fn"], "recall": ratio(agg["tp"], agg["tp"] + agg["fn"]),
                     "recall_lo": lo, "recall_hi": hi, "fp_frames": agg["fp"], "neg_frames": agg["fp"] + agg["tn"],
                     "box_precision": ratio(agg["tp"], agg["tp"] + agg["box_fp"]),
                     "mean_iou": float(np.mean(ious)) if ious else math.nan,
                     "mean_age": float(np.mean(ages)), "max_age": int(np.max(ages)),
                     "entry_latency": entry_lat, "exit_latency": exit_lat, "est_s": agg["ms"] / 1000.0})
    df = pd.DataFrame(rows)
    base = float(df.loc[df.family == "every", "est_s"].iloc[0])
    df["speedup"] = base / df["est_s"]
    return df, pd.DataFrame(per_scenario)


def interpolate_fixed(policies, target_ratio, column="recall"):
    fixed = policies[policies.family == "fixed"].sort_values("ratio")
    return float(np.interp(target_ratio, fixed["ratio"], fixed[column]))


