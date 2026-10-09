"""Figure generation for the benchmark."""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from motiongate import GateConfig, make_gate, theory

from . import synthetic as syn
from .settings import COLOR, GATES, LABEL, REFRESH, ROOT, SHORT, SWEEP_SCALES

def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def fig_envelope(agg, scene, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    panels = [("noise", r"noise $\sigma$ (grey levels)", "(a) Sensor noise, empty scene"),
              ("illumination", r"intensity step $\Delta$ (grey levels)", "(b) Illumination steps, empty scene"),
              ("jitter", "inter-frame translation (px)", "(c) Camera jitter, empty scene")]
    for ax, (kind, xlabel, title) in zip(axes.ravel()[:3], panels):
        for g in ("difference", "stabilized"):
            s = agg[(agg.kind == kind) & (agg.gate == g)].sort_values("value")
            ax.plot(s.value, s.rate, marker="o", ms=3, color=COLOR[g], label=LABEL[g])
            ax.fill_between(s.value, s.lo, s.hi, color=COLOR[g], alpha=0.15, lw=0)
        ax.set(xlabel=xlabel, ylabel="gate activation rate", title=title, ylim=(-0.03, 1.03))
    ax = axes[0, 0]
    pc = theory.percolation_probability(cfg.dilation_iterations)
    for g, pipe in (("difference", "classic"), ("stabilized", "robust")):
        lo, hi = (theory.critical_sigma(cfg.threshold, p, pipe) for p in (1e-2, 1e-1))
        ax.axvspan(lo, hi, color=COLOR[g], alpha=0.12, lw=0)
        ax.axvline(theory.critical_sigma(cfg.threshold, pc, pipe), color=COLOR[g], ls=":", lw=1)
    ax.text(0.98, 0.35, "bands: predicted onset\n(dotted: percolation $p_c$)", transform=ax.transAxes,
            ha="right", fontsize=6.5)
    axes[0, 1].axvline(theory.critical_illumination_step(cfg.threshold), color=COLOR["difference"], ls="--", lw=1)
    axes[0, 1].text(cfg.threshold + 2, 0.5, r"$\Delta=\tau$", color=COLOR["difference"], fontsize=8)
    axes[0, 0].legend(loc="upper left", frameon=False)
    ax = axes[1, 1]
    for scale, colour in zip(SWEEP_SCALES, ("#CC79A7", "#E69F00", "#56B4E9")):
        h = int(454 * scale)
        for g, ls in (("difference", "-"), ("stabilized", "--")):
            s = agg[(agg.kind == "speed") & (agg.gate == g) & np.isclose(agg.scale.astype(float), scale)].sort_values("value")
            ax.plot(s.value, s.rate, ls=ls, marker="o", ms=3, color=colour,
                    label=f"h={h} px, {SHORT[g].replace('FDS', 'FD-S')}")
        vstar = theory.area_model_critical_speed(h, cfg.min_area, cfg.dilation_iterations)
        if vstar > 0:
            ax.axvline(vstar, color=colour, lw=0.8, ls=":")
    ax.set(xlabel="person displacement (px/frame)", ylabel="gate recall", title="(d) Walking person, by height",
           ylim=(-0.03, 1.03))
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_envelope")


def fig_tradeoff(policies, video):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    fx = policies[policies.family == "fixed"].sort_values("ratio")
    ev = policies[policies.family == "every"].iloc[0]
    panels = ((axes[0], "recall", "frame recall (IoU $\\geq$ 0.5)", "(a) Controlled: recall"),
              (axes[1], "mean_iou", "mean IoU of hits", "(b) Controlled: localisation"))
    for ax, col, ylabel, title in panels:
        ax.plot(fx.ratio, fx[col], marker="s", ms=3, color=COLOR["fixed"], label="fixed rate, $N$ varied")
        for g in GATES:
            sub = policies[(policies.family == "gated") & (policies.gate == g) & (policies.K == REFRESH)].sort_values("ratio")
            ax.plot(sub.ratio, sub[col], marker="o", ms=3.2, color=COLOR[g], label=f"{LABEL[g]}, $M$ varied")
        ax.plot([1.0], [ev[col]], marker="*", ms=8, ls="none", color=COLOR["every"], label="every frame")
        ax.set(xlabel="invocation ratio $r$", ylabel=ylabel, title=title, xlim=(0, 1.04))
    axes[0].legend(loc="lower right", frameon=False, fontsize=5.8)
    ax = axes[2]
    if video is not None:
        v = video[0]
        vf = v[v.family == "fixed"].sort_values("ratio")
        ax.plot(vf.ratio, vf.recall, marker="s", ms=3, color=COLOR["fixed"])
        for g in GATES:
            sub = v[(v.family == "gated") & (v.gate == g)].sort_values("ratio")
            ax.plot(sub.ratio, sub.recall, marker="o", ms=3.2, color=COLOR[g])
        ax.set(xlabel="invocation ratio $r$", ylabel="recall vs. every-frame YOLOv8m",
               title="(c) Real video", xlim=(0, 1.04))
    fig.tight_layout()
    save(fig, "fig_tradeoff")

def fig_scenarios(per_scenario, scenarios):
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    x = np.arange(len(scenarios))
    width = 0.27
    for i, g in enumerate(GATES):
        s = per_scenario[per_scenario.policy == g].set_index("scenario").loc[[sc.name for sc in scenarios]]
        ax.bar(x + (i - 1) * width, s.ratio, width, color=COLOR[g], label=LABEL[g])
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(1.0 / REFRESH, color="k", lw=0.8, ls=":")
    ax.text(len(scenarios) - 0.45, 1.0 / REFRESH + 0.02, "1/K", fontsize=7, ha="right")
    ax.set_xticks(x, [sc.label for sc in scenarios], rotation=25, ha="right")
    ax.set(ylabel="invocation ratio r", ylim=(0, 1.08))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    save(fig, "fig_scenarios")


def fig_pipeline(args, scene, scenarios, seed0):
    idx = next(i for i, sc in enumerate(scenarios) if sc.name == "jitter_walk")
    frames = [r.frame for r in syn.render_scenario(scene, scenarios[idx], syn.scenario_rng(args.seed, idx))]
    t = 40
    cfg = GateConfig(keep_mask=True)
    fd, fds = make_gate("difference", cfg), make_gate("stabilized", cfg)
    fd.update(frames[t - 1])
    fds.update(frames[t - 1])
    r_fd, r_fds = fd.update(frames[t]), fds.update(frames[t])
    out = frames[t].copy()
    for x1, y1, x2, y2 in r_fds.boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 170, 0), 2)
    for b in seed0["jitter_walk"]["dets"][t]:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 3)
    panels = [(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB), "(a) frame $t$ (camera jitter)"),
              (r_fd.mask, f"(b) FD mask: {'active' if r_fd.active else 'inactive'}"),
              (r_fds.mask, f"(c) FD-S mask: {'active' if r_fds.active else 'inactive'}"),
              (cv2.cvtColor(out, cv2.COLOR_BGR2RGB), "(d) FD-S region (green), YOLO (red)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_pipeline")
    return r_fds.shift




def fig_tracking_comparison(summary):
    """Compare the paper baseline with optical-flow box propagation."""
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45))
    for mode, marker, label in (
        ("retain", "s", "retain last box"),
        ("flow", "o", "optical-flow tracking"),
    ):
        sub = summary[summary.mode == mode].sort_values("ratio")
        axes[0].plot(sub.ratio, sub.recall, marker=marker, ms=4, label=label)
        axes[1].plot(sub.ratio, sub.mean_iou, marker=marker, ms=4, label=label)
        axes[2].plot(sub.ratio, sub.speedup, marker=marker, ms=4, label=label)

    axes[0].set(
        xlabel="detector invocation ratio $r$",
        ylabel="frame recall",
        title="(a) Detection recall",
        xlim=(0, 1.04),
        ylim=(0, 1.01),
    )
    axes[1].set(
        xlabel="detector invocation ratio $r$",
        ylabel="mean IoU of hits",
        title="(b) Localisation",
        xlim=(0, 1.04),
        ylim=(0, 1.01),
    )
    axes[2].set(
        xlabel="detector invocation ratio $r$",
        ylabel="estimated speed-up",
        title="(c) Compute",
        xlim=(0, 1.04),
    )
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_tracking_comparison")
