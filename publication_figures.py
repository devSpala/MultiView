"""
publication_figures.py — final paper-ready figure suite
=========================================================
Clean, descriptive labels throughout (no internal code names like
"V5 (old)", "P3", "Homog+a", "colab"). Every label is something a
reviewer should see. Plus three upgraded chart types that convey more
information than simple bars.

  python publication_figures.py                 # all figures
  python publication_figures.py --errorbars off # no error bars

Label policy
------------
  Proposed (P3_fidelity_lg)  -> "Proposed"
  V5_full                    -> "Aggressive sharpening"   (the failed
                                 over-sharpened pipeline; descriptive,
                                 not a version code)
  B2_homog_alpha             -> "Homography + blend"
  B3_affine_alpha            -> "Affine + blend"
  Base_M1                    -> "Degraded input"
  runtime tag "colab"        -> "Client workstation"

Figures
-------
  fig_metrics        4-metric grouped bars, clamped error bars
  fig_ssim_box       SSIM distribution box plots per scenario
  fig_lapvar_box     edge-contrast box plots (log y) — skew-honest
  fig_radar          normalized multi-metric profile
  fig_alignment      registration success: classical vs learned
  fig_heatmap        per-scenario % change vs input, all metrics
  fig_runtime        per-stage latency
  fig_tradeoff       NEW: fidelity-vs-sharpness frontier (scatter,
                     every method as a point — shows the design space)
  fig_winrate        NEW: per-pair win-rate vs input (dual metric)
  fig_quality_gain   NEW: paired before/after slope chart (input->
                     proposed) for SSIM and NIQE
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "savefig.dpi": 300, "figure.dpi": 150,
})

# Clean colour + label registry — single source of truth.
C_INPUT, C_HOMO, C_AFF, C_PROP, C_OVER = \
    "#4C72B0", "#DD8452", "#937860", "#2E8B57", "#C44E52"
PROPOSED = "P3_fidelity_lg"

LABELS = {
    "Base_M1": "Degraded input",
    "B1_lanczos": "Lanczos upscale",
    "B2_homog_alpha": "Homography + blend",
    "B3_affine_alpha": "Affine + blend",
    "V2_affine_only": "Affine + blend",
    "V3_affine_flow": "Affine + flow",
    "V4_affine_lab": "Fusion + sharpening",
    "V5_full": "Aggressive sharpening",
    "V5_full_nopost": "Fusion (no sharpening)",
    "P1_fidelity": "Fusion (no sharpening)",
    "P2_light_post": "Fusion + light sharpening",
    "A1_lightglue": "Learned matching",
    "A2_raft": "Learned flow",
    "A3_learned": "Learned matching + flow",
    "P3_fidelity_lg": "Proposed",
}
RUNTIME_TAGS = {"colab": "Commodity CPU",
                "workstation": "Client workstation",
                "rpi4": "Raspberry Pi 4 (edge)",
                "rpi4_EXAMPLE": "Raspberry Pi 4 (edge)"}
# Order rows by this priority (lower first); unknown tags go last.
RUNTIME_ORDER = {"workstation": 0, "colab": 1, "rpi4": 2}


def lab(method):
    return LABELS.get(method, method)


def load(p="results.json"):
    return json.load(open(p))


def index_pairs(data):
    idx = defaultdict(dict)
    for r in data["records"]:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    return idx


def aligned_keys(idx, ref=PROPOSED):
    return [k for k, v in idx.items() if v.get(ref, {}).get("aligned")]


def vals(idx, keys, method, metric):
    return np.array([idx[k][method][metric] for k in keys
                     if idx[k].get(method, {}).get(metric) is not None],
                    dtype=float)


def clamped_err(mean, std, lo=0.0, hi=None):
    low = mean - max(lo, mean - std)
    high = std if hi is None else min(hi - mean, std)
    return np.array([[low], [high]])


# ── fig_metrics ──────────────────────────────────────────────────────
def fig_metrics(idx, show_err=True):
    keys = aligned_keys(idx)
    methods = [("Base_M1", C_INPUT), ("B2_homog_alpha", C_HOMO),
               (PROPOSED, C_PROP)]
    metrics = [("ssim", "SSIM \u2191", 0, 1), ("psnr", "PSNR \u2191", 0, None),
               ("lpips", "LPIPS \u2193", 0, 1), ("niqe", "NIQE \u2193", 0, None)]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.7))
    for ax, (mk, mlabel, lo, hi) in zip(axes, metrics):
        ymax = 0
        for i, (m, c) in enumerate(methods):
            v = vals(idx, keys, m, mk)
            if len(v) == 0:
                continue
            mean, std = v.mean(), v.std(ddof=1)
            err = clamped_err(mean, std, lo, hi) if show_err else None
            ax.bar(i, mean, 0.7, yerr=err, color=c, alpha=0.9, zorder=3,
                   error_kw=dict(elinewidth=0.9, ecolor="#444", capsize=2.5,
                                 capthick=0.9))
            cap = mean + (err[1][0] if err is not None else 0)
            ax.text(i, cap, f" {mean:.2f}" if mk != "niqe" else f" {mean:.1f}",
                    ha="center", va="bottom", fontsize=6.4, fontweight="bold")
            ymax = max(ymax, cap)
        ax.axhline(vals(idx, keys, "Base_M1", mk).mean(), color=C_INPUT,
                   ls="--", lw=0.7, alpha=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([lab(m) for m, _ in methods], fontsize=5.6,
                           rotation=20, ha="right")
        ax.set_title(mlabel, fontsize=8.5)
        ax.set_ylim(0, ymax * 1.18)
    fig.suptitle("Reference and no-reference quality (aligned pairs; "
                 "dashed line = input level)", fontsize=8, y=1.05)
    fig.tight_layout(pad=0.5)
    fig.savefig("fig_metrics.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_metrics done")


# ── fig_ssim_box ─────────────────────────────────────────────────────
def fig_ssim_box(idx):
    scen = sorted({k[0] for k in idx})
    methods = [("Base_M1", C_INPUT), (PROPOSED, C_PROP)]
    fig, ax = plt.subplots(figsize=(6.0, 3.0)); w = 0.36
    for j, (m, c) in enumerate(methods):
        data = [vals(idx, [k for k in idx if k[0] == s
                           and idx[k].get(PROPOSED, {}).get("aligned")],
                     m, "ssim") for s in scen]
        pos = np.arange(len(scen)) + (j - 0.5) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.0))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.75)
    ax.axhline(0.85, color="#777", ls="--", lw=0.8)
    ax.text(len(scen) - 0.5, 0.855, "0.85 near-lossless", fontsize=6.3,
            color="#555")
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen)
    ax.set_ylabel("SSIM (distribution across pairs)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75) for _, c in methods]
    ax.legend(handles, [lab(m) for m, _ in methods], loc="lower left",
              framealpha=0.92)
    ax.set_title("Structural fidelity preserved across all scenarios",
                 fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_ssim_box.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_ssim_box done")


# ── fig_lapvar_box ───────────────────────────────────────────────────
def fig_lapvar_box(idx):
    keys = aligned_keys(idx)
    methods = [("Base_M1", C_INPUT), ("V5_full", C_OVER), (PROPOSED, C_PROP)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    data = [vals(idx, keys, m, "lapvar") for m, _ in methods]
    data = [d[d > 0] for d in data]
    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3,
                                    markerfacecolor="#999", alpha=0.5),
                    medianprops=dict(color="black", lw=1.2))
    for box, (m, c) in zip(bp["boxes"], methods):
        box.set(facecolor=c, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticklabels([lab(m) for m, _ in methods], fontsize=7)
    ax.set_ylabel("Edge contrast (Laplacian variance, log)")
    ax.set_title("Edge contrast: median + IQR.\nProposed lifts contrast "
                 "modestly; aggressive sharpening over-amplifies",
                 fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_lapvar_box.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_lapvar_box done")


# ── fig_radar ────────────────────────────────────────────────────────
def fig_radar(idx):
    keys = aligned_keys(idx)
    specs = [("SSIM", "ssim", False), ("PSNR", "psnr", False),
             ("LPIPS", "lpips", True), ("NIQE", "niqe", True),
             ("Edge\ncontrast", "lapvar", False)]
    methods = [("Base_M1", C_INPUT), ("V5_full", C_OVER), (PROPOSED, C_PROP)]
    raw = {m: [vals(idx, keys, m, mk).mean() for _, mk, _ in specs]
           for m, _ in methods}
    norm = {m: [] for m, _ in methods}
    for i, (_, _, lower_better) in enumerate(specs):
        col = [raw[m][i] for m, _ in methods]
        lo, hi = min(col), max(col)
        for m, _ in methods:
            x = (raw[m][i] - lo) / (hi - lo + 1e-9)
            norm[m].append(1 - x if lower_better else x)
    ang = np.linspace(0, 2 * np.pi, len(specs), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.8, 4.6), subplot_kw=dict(polar=True))
    for m, c in methods:
        d = norm[m] + norm[m][:1]
        ax.plot(ang, d, color=c, lw=1.8, label=lab(m), zorder=3)
        ax.fill(ang, d, color=c, alpha=0.12, zorder=2)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([s for s, _, _ in specs], fontsize=7.5)
    ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.13), framealpha=0.92)
    ax.set_title("Multi-metric profile (each axis scaled across the three\n"
                 "methods shown; outer = better)", fontsize=7.8, pad=20)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_radar.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_radar done")


# ── fig_alignment ────────────────────────────────────────────────────
def fig_alignment(idx):
    scen = sorted({k[0] for k in idx})
    classical, learned = [], []
    for s in scen:
        ks = [k for k in idx if k[0] == s]
        classical.append(sum(1 for k in ks
                             if idx[k].get("V5_full", {}).get("aligned")) / len(ks) * 100)
        learned.append(sum(1 for k in ks
                           if idx[k].get(PROPOSED, {}).get("aligned")) / len(ks) * 100)
    x = np.arange(len(scen)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.bar(x - w/2, classical, w, label="Classical matching (SIFT)",
           color=C_OVER, alpha=0.9, zorder=3)
    ax.bar(x + w/2, learned, w, label="Learned matching (proposed)",
           color=C_PROP, alpha=0.9, zorder=3)
    for i, (a, b) in enumerate(zip(classical, learned)):
        ax.text(i - w/2, a + 2, f"{a:.0f}", ha="center", fontsize=6.3)
        ax.text(i + w/2, b + 2, f"{b:.0f}", ha="center", fontsize=6.3)
    ax.set_xticks(x); ax.set_xticklabels(scen); ax.set_ylim(0, 112)
    ax.set_ylabel("Registration success (%)")
    ax.legend(loc="lower center", ncol=2, framealpha=0.92)
    ax.set_title("Learned matching achieves full registration success",
                 fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_alignment.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_alignment done")


# ── fig_heatmap ──────────────────────────────────────────────────────
def fig_heatmap(idx):
    scen = sorted({k[0] for k in idx})
    specs = [("SSIM", "ssim", False), ("PSNR", "psnr", False),
             ("LPIPS", "lpips", True), ("NIQE", "niqe", True)]
    M = np.zeros((len(specs), len(scen)))
    for j, s in enumerate(scen):
        keys = [k for k in idx if k[0] == s
                and idx[k].get(PROPOSED, {}).get("aligned")]
        for i, (_, mk, lower_better) in enumerate(specs):
            b = vals(idx, keys, "Base_M1", mk).mean()
            p = vals(idx, keys, PROPOSED, mk).mean()
            pct = (p - b) / abs(b) * 100
            M[i, j] = -pct if lower_better else pct
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen)
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([s for s, _, _ in specs])
    for i in range(len(specs)):
        for j in range(len(scen)):
            ax.text(j, i, f"{M[i, j]:+.0f}%", ha="center", va="center",
                    fontsize=6.5, color="#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% improvement over input", fontsize=6.5)
    ax.set_title("Proposed vs degraded input, per scenario\n"
                 "(green = improvement)", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_heatmap.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_heatmap done")


# ── fig_runtime ──────────────────────────────────────────────────────
def fig_runtime(benches):
    stage_lab = {"s2_registration_ms": "Registration",
                 "s3_flow_ms": "Dense flow", "s4s5_fusion_ms": "Fusion",
                 "s5_post_ms": "Post-processing"}
    palette = ["#4C72B0", "#DD8452", "#2E8B57", "#937860"]
    benches = sorted(benches, key=lambda b: RUNTIME_ORDER.get(b["tag"], 99))
    keys = [k for k in stage_lab if any(k in b["stages"] for b in benches)]
    n = len(benches)
    # Taller base so a short (1-row) chart still has room beneath the axis
    # for BOTH the x-axis label and the legend without them overlapping.
    fig, ax = plt.subplots(figsize=(5.8, 2.4 + 0.55 * n))
    for yi, b in enumerate(benches):
        left = 0
        for ki, k in enumerate(keys):
            v = b["stages"].get(k, {}).get("mean", 0)
            ax.barh(yi, v, 0.5, left=left, color=palette[ki], alpha=0.9, zorder=3)
            if v > 0.05 * b["total"]["mean"]:
                ax.text(left + v/2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
            left += v
        ax.text(left * 1.01, yi, f"{b['total']['mean']:.0f} ms",
                va="center", fontsize=6.8, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([RUNTIME_TAGS.get(b["tag"], b["tag"]) for b in benches])
    ax.set_xlabel("Per-stage latency (ms)")
    ax.set_xlim(0, max(b["total"]["mean"] for b in benches) * 1.3)
    ax.set_title("Per-stage processing latency", fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[i]) for i in range(len(keys))]
    # Reserve a generous band at the bottom for the x-axis label AND the
    # legend, then place the legend below the label inside that band.
    # Reserving the space is what stops a tight bbox from pulling the
    # legend back up onto the axis (the original overlap bug).
    fig.subplots_adjust(bottom=0.40, top=0.84, left=0.22, right=0.97)
    fig.legend(handles, [stage_lab[k] for k in keys], loc="upper center",
               ncol=len(keys), fontsize=6.4,
               bbox_to_anchor=(0.5, 0.25), frameon=True, framealpha=0.92)

    # No tight_layout (it overrides subplots_adjust) and no
    # bbox_inches="tight" (it re-crops and can re-introduce the overlap).
    fig.savefig("fig_runtime.png", dpi=300)
    plt.close(fig)
    print("  fig_runtime done")


# ── fig_tradeoff (NEW) — the design-space frontier ───────────────────
def fig_tradeoff(idx):
    """Every method as a point in (edge-contrast, fidelity) space. Shows
    at a glance which methods buy sharpness at the cost of fidelity and
    which sit on the good frontier."""
    keys = aligned_keys(idx)
    plot = [("Base_M1", C_INPUT, "o"), ("B2_homog_alpha", C_HOMO, "^"),
            ("V3_affine_flow", C_AFF, "v"), ("V5_full", C_OVER, "s"),
            ("V4_affine_lab", "#c9a86a", "D"), (PROPOSED, C_PROP, "*")]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    base_ssim = vals(idx, keys, "Base_M1", "ssim").mean()
    for m, c, mk in plot:
        lv = vals(idx, keys, m, "lapvar").mean()
        ss = vals(idx, keys, m, "ssim").mean()
        size = 230 if m == PROPOSED else 90
        ax.scatter(lv, ss, c=c, marker=mk, s=size, alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=4, label=lab(m))
    ax.axhline(base_ssim, color=C_INPUT, ls="--", lw=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1], base_ssim + 0.004, "input fidelity",
            ha="right", fontsize=6.3, color=C_INPUT)
    ax.set_xscale("log")
    ax.set_xlabel("Edge contrast (Laplacian variance, log) \u2192 sharper")
    ax.set_ylabel("SSIM \u2192 more faithful")
    ax.legend(loc="lower left", framealpha=0.92, fontsize=6.3)
    ax.set_title("Fidelity-vs-sharpness design space\n"
                 "(top-right is ideal; proposed keeps fidelity while "
                 "adding contrast)", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig_tradeoff.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_tradeoff done")


# ── fig_winrate (NEW) ────────────────────────────────────────────────
def fig_winrate(idx):
    keys_all = list(idx.keys())
    methods = [PROPOSED, "P1_fidelity", "P2_light_post", "V5_full"]
    fig, ax = plt.subplots(figsize=(5.6, 2.7)); yy = np.arange(len(methods)); w = 0.38
    for metric, low, off, col, leg in [
            ("ssim", False, -w/2, C_PROP, "beats input on SSIM"),
            ("lpips", True, w/2, C_INPUT, "beats input on LPIPS")]:
        rates = []
        for m in methods:
            wins = tot = 0
            for k in keys_all:
                b = idx[k].get("Base_M1", {}).get(metric)
                v = idx[k].get(m, {}).get(metric)
                if b is None or v is None:
                    continue
                tot += 1
                if (v < b) if low else (v > b):
                    wins += 1
            rates.append(wins / tot * 100 if tot else 0)
        ax.barh(yy + off, rates, w, color=col, alpha=0.9, zorder=3, label=leg)
        for i, r in enumerate(rates):
            ax.text(r + 1, yy[i] + off, f"{r:.0f}%", va="center", fontsize=6.2)
    ax.axvline(50, color="#777", ls="--", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([lab(m) for m in methods])
    ax.set_xlabel("% of pairs that beat the degraded input"); ax.set_xlim(0, 100)
    ax.set_title("Win-rate against the degraded input", fontsize=8.5)
    # Lift the legend OUT of the plotting area (every corner has bars, so an
    # in-axes legend overlaps them). Place it above the axes, with reserved
    # space so a tight bbox can't pull it back down onto the bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
              framealpha=0.92, fontsize=7)
    fig.subplots_adjust(top=0.82, left=0.30, right=0.97, bottom=0.16)
    fig.savefig("fig_winrate.png", dpi=300); plt.close(fig)
    print("  fig_winrate done")


# ── fig_quality_gain (NEW) — paired slope chart ──────────────────────
def fig_quality_gain(idx):
    """Slope chart: each pair's value at input vs proposed, for two
    complementary metrics. Shows direction & consistency of change."""
    keys = aligned_keys(idx)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 3.2))
    for ax, metric, title, better_up in [
            (a1, "ssim", "SSIM (higher better)", True),
            (a2, "niqe", "NIQE (lower better)", False)]:
        for k in keys:
            b = idx[k].get("Base_M1", {}).get(metric)
            p = idx[k].get(PROPOSED, {}).get(metric)
            if b is None or p is None:
                continue
            improved = (p > b) if better_up else (p < b)
            ax.plot([0, 1], [b, p], color=(C_PROP if improved else C_OVER),
                    alpha=0.35, lw=0.8, zorder=2)
        mb = vals(idx, keys, "Base_M1", metric).mean()
        mp = vals(idx, keys, PROPOSED, metric).mean()
        ax.plot([0, 1], [mb, mp], color="black", lw=2.2, zorder=4,
                marker="o", markersize=5)
        ax.text(0, mb, f" {mb:.2f}", ha="right", fontsize=7, fontweight="bold")
        ax.text(1, mp, f" {mp:.2f}", ha="left", fontsize=7, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Degraded\ninput", "Proposed"], fontsize=7)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, fontsize=8)
    fig.suptitle("Per-pair change from input to proposed "
                 "(green = improved, red = worse)", fontsize=8, y=1.02)
    fig.tight_layout(pad=0.5)
    fig.savefig("fig_quality_gain.png", bbox_inches="tight"); plt.close(fig)
    print("  fig_quality_gain done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errorbars", choices=["on", "off"], default="on")
    ap.add_argument("--results", default="results.json")
    args = ap.parse_args()
    idx = index_pairs(load(args.results))
    print("Generating publication figures...")
    fig_metrics(idx, show_err=(args.errorbars == "on"))
    fig_ssim_box(idx)
    fig_lapvar_box(idx)
    fig_radar(idx)
    fig_alignment(idx)
    fig_heatmap(idx)
    fig_tradeoff(idx)
    fig_winrate(idx)
    fig_quality_gain(idx)
    benches = [json.load(open(p)) for p in sorted(glob.glob("runtime_*.json"))]
    if benches:
        fig_runtime(benches)
    else:
        print("  fig_runtime skipped (no runtime_*.json)")
    print("Done.")


if __name__ == "__main__":
    main()
