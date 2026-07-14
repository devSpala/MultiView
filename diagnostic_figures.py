"""
diagnostic_figures.py — honest analysis figures for the REAL dataset
=====================================================================
The standard 7 figures assume the method wins. Your real data shows it
does not (yet): post-processing destroys SSIM and 37% of pairs fail to
align. These diagnostic figures are built to SURFACE those problems so
you can fix the method and tell an honest story — they are analysis
tools, not paper figures to submit as-is.

Reads results.json + sensitivity.json (from run_experiments / run_sweep).

  python diagnostic_figures.py            # writes diag1..diag7 .png

The seven diagnostics
---------------------
  diag1_alignment_health   alignment success rate per scenario + inlier
                           distribution (shows the registration crisis)
  diag2_ssim_waterfall     SSIM stage-by-stage on ALIGNED pairs only
                           (pinpoints which stage destroys fidelity)
  diag3_aligned_vs_all     how much the failed pairs distort the means
  diag4_ssim_lapvar_tradeoff  per-pair scatter: the fidelity/sharpness
                           conflict, with base reference lines
  diag5_sensitivity_truth  the sweep, framed honestly: more sharpening
                           = lower SSIM (the knob makes it worse)
  diag6_per_scenario_ssim  which scenarios the method survives / fails
  diag7_method_ranking     paired win-rate vs base, per method
"""

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

C_BASE, C_BAD, C_OK, C_PROP, C_WARN = \
    "#4C72B0", "#C44E52", "#55A868", "#2E8B57", "#DD8452"


def load(p="results.json"):
    return json.load(open(p))


def index_pairs(data):
    idx = defaultdict(dict)
    for r in data["records"]:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    return idx


def aligned_keys(idx, ref="V5_full"):
    return [k for k, v in idx.items() if v.get(ref, {}).get("aligned")]


STAGE_ORDER = [
    ("Base_M1", "Base\n$M_1$", C_BASE),
    ("V2_affine_only", "V2\nAffine\n+Alpha", C_WARN),
    ("V3_affine_flow", "V3\n+Flow", C_WARN),
    ("V5_full_nopost", "V5⁻\nLAB\nfusion", C_OK),
    ("V4_affine_lab", "V4\nLAB\n+post", C_BAD),
    ("V5_full", "V5\nFull", C_PROP),
]


# ── diag1: alignment health ──────────────────────────────────────────
def diag1_alignment(data, idx):
    scen = sorted({k[0] for k in idx})
    succ, tot, inliers = [], [], []
    for s in scen:
        keys = [k for k in idx if k[0] == s]
        a = sum(1 for k in keys if idx[k].get("V5_full", {}).get("aligned"))
        succ.append(a); tot.append(len(keys))
    for k in idx:
        r = idx[k].get("V5_full", {})
        if r.get("aligned") and r.get("n_inliers") is not None:
            inliers.append(r["n_inliers"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    rate = [s / t * 100 for s, t in zip(succ, tot)]
    cols = [C_OK if r >= 80 else C_WARN if r >= 50 else C_BAD for r in rate]
    a1.bar(range(len(scen)), rate, color=cols, alpha=0.9, zorder=3)
    for i, (r, s, t) in enumerate(zip(rate, succ, tot)):
        a1.text(i, r + 2, f"{s}/{t}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")
    a1.axhline(80, color="#777", ls="--", lw=0.8)
    a1.set_xticks(range(len(scen))); a1.set_xticklabels(scen)
    a1.set_ylabel("Alignment success (%)"); a1.set_ylim(0, 108)
    a1.set_title("(a) Registration failures by scenario", fontsize=8)

    a2.hist(inliers, bins=range(0, max(inliers) + 20, 10),
            color=C_BASE, alpha=0.85, zorder=3, edgecolor="white")
    a2.axvline(15, color=C_BAD, ls="--", lw=1.0)
    a2.text(16, a2.get_ylim()[1] * 0.9, "15 = fragility\nthreshold",
            fontsize=6.5, color=C_BAD, va="top")
    a2.set_xlabel("RANSAC inliers (successful pairs)")
    a2.set_ylabel("Count")
    a2.set_title("(b) Inlier counts: dangerously low", fontsize=8)
    med = int(np.median(inliers))
    a2.text(0.97, 0.6, f"median = {med}\n(healthy: >50)",
            transform=a2.transAxes, ha="right", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6",
                      ec="#dca", lw=0.7))
    fig.tight_layout(pad=0.5)
    fig.savefig("diag1_alignment_health.png", bbox_inches="tight")
    plt.close(fig); print("  diag1_alignment_health done")


# ── diag2: SSIM waterfall (aligned only) ─────────────────────────────
def diag2_waterfall(data, idx):
    keys = aligned_keys(idx)
    def mean(m, key="ssim"):
        v = [idx[k][m][key] for k in keys
             if idx[k].get(m, {}).get(key) is not None]
        return np.mean(v) if v else np.nan
    labels, cols, vals = [], [], []
    for m, lab, c in STAGE_ORDER:
        labels.append(lab); cols.append(c); vals.append(mean(m))
    base = vals[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, alpha=0.9, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")
    ax.axhline(base, color=C_BASE, ls="--", lw=0.9, alpha=0.7)
    ax.text(len(labels) - 0.4, base + 0.004, "base level",
            fontsize=6.5, color=C_BASE, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("SSIM vs ground truth (aligned pairs)")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.05)
    ax.set_title("Where structural fidelity is lost\n"
                 "LAB fusion (V5⁻) is safe; post-processing destroys it",
                 fontsize=8)
    ax.annotate("", xy=(4, vals[4] + 0.02), xytext=(3, vals[3] + 0.02),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.4))
    ax.text(3.5, max(vals[3], vals[4]) + 0.035,
            f"−{vals[3] - vals[4]:.3f}\n(post-proc.)",
            ha="center", fontsize=6.8, color=C_BAD, fontweight="bold")
    fig.tight_layout(pad=0.4)
    fig.savefig("diag2_ssim_waterfall.png", bbox_inches="tight")
    plt.close(fig); print("  diag2_ssim_waterfall done")


# ── diag3: aligned-only vs all-pairs means ───────────────────────────
def diag3_aligned_vs_all(data, idx):
    methods = ["Base_M1", "V5_full_nopost", "V5_full", "A2_raft"]
    keys_all = list(idx.keys())
    keys_al = aligned_keys(idx)
    def mean(keys, m):
        v = [idx[k][m]["ssim"] for k in keys
             if idx[k].get(m, {}).get("ssim") is not None
             and (m == "Base_M1" or idx[k][m].get("aligned")
                  or keys is keys_all)]
        return np.mean(v) if v else np.nan
    all_m = [mean(keys_all, m) for m in methods]
    al_m = [mean(keys_al, m) for m in methods]
    x = np.arange(len(methods)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.bar(x - w/2, all_m, w, label="All pairs (incl. failures)",
           color=C_WARN, alpha=0.9, zorder=3)
    ax.bar(x + w/2, al_m, w, label="Aligned pairs only",
           color=C_OK, alpha=0.9, zorder=3)
    for i, (a, b) in enumerate(zip(all_m, al_m)):
        ax.text(i - w/2, a + 0.006, f"{a:.3f}", ha="center", va="bottom",
                fontsize=6.2)
        ax.text(i + w/2, b + 0.006, f"{b:.3f}", ha="center", va="bottom",
                fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "V5⁻", "V5", "A2 RAFT"], fontsize=7)
    ax.set_ylabel("Mean SSIM"); ax.set_ylim(0.5, 0.9)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_title("Failed pairs inflate the means — report aligned-only",
                 fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("diag3_aligned_vs_all.png", bbox_inches="tight")
    plt.close(fig); print("  diag3_aligned_vs_all done")


# ── diag4: per-pair SSIM vs LapVar tradeoff ──────────────────────────
def diag4_tradeoff(data, idx):
    keys = aligned_keys(idx)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, lab, c, mk in [("V5_full_nopost", "V5⁻ (LAB only)", C_OK, "o"),
                          ("V5_full", "V5 (full)", C_BAD, "s")]:
        pts2 = [(idx[k][m]["lapvar"], idx[k][m]["ssim"]) for k in keys
                if idx[k].get(m, {}).get("ssim") is not None]
        xs = [p[0] for p in pts2]; ys = [p[1] for p in pts2]
        ax.scatter(xs, ys, c=c, label=lab, s=22, alpha=0.7,
                   edgecolors="white", linewidths=0.4, zorder=3, marker=mk)
    bvals = [idx[k]["Base_M1"]["ssim"] for k in keys
             if idx[k].get("Base_M1", {}).get("ssim") is not None]
    bs = np.mean(bvals)
    ax.axhline(bs, color=C_BASE, ls="--", lw=0.9)
    ax.text(ax.get_xlim()[1], bs + 0.004, "base SSIM",
            ha="right", fontsize=6.5, color=C_BASE)
    ax.set_xscale("log")
    ax.set_xlabel("Laplacian variance (log) — edge contrast")
    ax.set_ylabel("SSIM vs ground truth")
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title("The conflict: sharpening up, fidelity down\n"
                 "points should sit ABOVE the base line — they don't",
                 fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("diag4_ssim_lapvar_tradeoff.png", bbox_inches="tight")
    plt.close(fig); print("  diag4_ssim_lapvar_tradeoff done")


# ── diag5: sensitivity, honest framing ───────────────────────────────
def diag5_sensitivity(sens):
    pts = sens["points"]
    a = np.array([p["alpha"] for p in pts])
    lap = np.array([p["lapvar"]["mean"] for p in pts])
    ss = np.array([p["ssim"]["mean"] for p in pts])
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    ax1.plot(a, ss, "o-", color=C_BASE, markersize=6, zorder=4,
             label="SSIM (fidelity)")
    ax1.set_xlabel(r"Unsharp strength $\alpha$")
    ax1.set_ylabel("SSIM", color=C_BASE)
    ax1.tick_params(axis="y", labelcolor=C_BASE)
    ax2 = ax1.twinx(); ax2.spines.right.set_visible(True)
    ax2.plot(a, lap, "s--", color=C_BAD, markersize=6, zorder=4,
             label="Laplacian var.")
    ax2.set_ylabel("Laplacian variance", color=C_BAD)
    ax2.tick_params(axis="y", labelcolor=C_BAD); ax2.grid(False)
    ax1.set_title("Sensitivity sweep tells the truth:\n"
                  "every increase in sharpening LOWERS fidelity",
                  fontsize=8)
    ax1.annotate("lower α = higher fidelity",
                 xy=(a[0], ss[0]), xytext=(a[0] + 0.25, ss[0] - 0.02),
                 fontsize=6.8, color=C_OK,
                 arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.0))
    fig.tight_layout(pad=0.4)
    fig.savefig("diag5_sensitivity_truth.png", bbox_inches="tight")
    plt.close(fig); print("  diag5_sensitivity_truth done")


# ── diag6: per-scenario SSIM (aligned) ───────────────────────────────
def diag6_per_scenario(data, idx):
    scen = sorted({k[0] for k in idx})
    methods = [("Base_M1", C_BASE, "o"), ("V5_full_nopost", C_OK, "^"),
               ("V5_full", C_BAD, "s")]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(len(scen))
    for m, c, mk in methods:
        ys = []
        for s in scen:
            keys = [k for k in idx if k[0] == s
                    and idx[k].get("V5_full", {}).get("aligned")]
            v = [idx[k][m]["ssim"] for k in keys
                 if idx[k].get(m, {}).get("ssim") is not None]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot(x, ys, marker=mk, color=c, markersize=6,
                label={"Base_M1": "Base", "V5_full_nopost": "V5⁻ (LAB)",
                       "V5_full": "V5 (full)"}[m], zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scen)
    ax.set_ylabel("Mean SSIM (aligned pairs)")
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title("Per-scenario fidelity: V5⁻ tracks base, V5 falls below",
                 fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("diag6_per_scenario_ssim.png", bbox_inches="tight")
    plt.close(fig); print("  diag6_per_scenario_ssim done")


# ── diag7: win-rate vs base ──────────────────────────────────────────
def diag7_winrate(data, idx):
    keys = aligned_keys(idx)
    methods = ["V5_full_nopost", "V4_affine_lab", "V5_full", "A2_raft",
               "B2_homog_alpha"]
    labels = ["V5⁻ LAB", "V4 LAB+post", "V5 full", "A2 RAFT", "B2 Homog"]
    rates = []
    for m in methods:
        wins = tot = 0
        for k in keys:
            b = idx[k]["Base_M1"].get("ssim")
            v = idx[k].get(m, {}).get("ssim")
            if b is None or v is None:
                continue
            tot += 1; wins += 1 if v > b else 0
        rates.append(wins / tot * 100 if tot else 0)
    cols = [C_OK if r >= 50 else C_BAD for r in rates]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.barh(range(len(methods)), rates, color=cols, alpha=0.9, zorder=3)
    for i, r in enumerate(rates):
        ax.text(r + 1, i, f"{r:.0f}%", va="center", fontsize=7,
                fontweight="bold")
    ax.axvline(50, color="#777", ls="--", lw=0.8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(labels)
    ax.set_xlabel("% of aligned pairs that BEAT base SSIM")
    ax.set_xlim(0, 100)
    ax.set_title("Win-rate vs base: only LAB-fusion-without-post wins",
                 fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("diag7_method_ranking.png", bbox_inches="tight")
    plt.close(fig); print("  diag7_method_ranking done")


def main():
    data = load("results.json")
    idx = index_pairs(data)
    print("Generating diagnostic figures...")
    diag1_alignment(data, idx)
    diag2_waterfall(data, idx)
    diag3_aligned_vs_all(data, idx)
    diag4_tradeoff(data, idx)
    try:
        diag5_sensitivity(load("sensitivity.json"))
    except FileNotFoundError:
        print("  diag5 skipped (no sensitivity.json)")
    diag6_per_scenario(data, idx)
    diag7_winrate(data, idx)
    print("Done. These are ANALYSIS figures — fix the method, then "
          "regenerate the paper figures.")


if __name__ == "__main__":
    main()
