"""
refsr_figures.py — Route A add-on to publication_figures.py
============================================================
Generates the THREE new RefSR-comparison figures from the matched-
resolution run (compare_proc512.json). Kept separate from
publication_figures.py because that script reads results.json (full-res,
method keys Base_M1/V5_full/P3_fidelity_lg) whereas these read the
matched-resolution comparison (keys Input/Proposed/Proposed (single)/
TTSR-rec). Same fonts/colours/DPI as your main script.

Produces:
  fig_ocr_3way.png        -> REPLACES fig_ocr_v2.png (honest matched OCR)
  fig_refsr_tradeoff.png  -> NEW centrepiece: fidelity vs readability
  fig_refsr_metrics.png   -> NEW 4-panel SSIM/LPIPS/NIQE/OCR comparison

USAGE:
  python refsr_figures.py                       # uses compare_proc512.json
  python refsr_figures.py --results compare_proc512.json --errorbars off
"""

import argparse
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

# match the main script's palette where possible
C_INPUT = "#777777"
C_PROP = "#2E8B57"          # proposed = same green family as main script
C_SINGLE = "#7FCDBB"
C_TTSR = "#C44E52"          # same red as C_OVER in main script

METHODS = ["Input", "Proposed", "Proposed (single)", "TTSR-rec"]
LBL = {"Input": "Input", "Proposed": "Proposed\n(multiscale)",
       "Proposed (single)": "Proposed\n(single)", "TTSR-rec": "TTSR-rec"}
COL = {"Input": C_INPUT, "Proposed": C_PROP,
       "Proposed (single)": C_SINGLE, "TTSR-rec": C_TTSR}


def load(p):
    return json.load(open(p))


def summ(data):
    return data["summary"]


def mean_of(s, m, k):
    return s[m][k]["mean"] if s.get(m, {}).get(k) else None


def sem_of(s, m, k):
    v = s.get(m, {}).get(k)
    return v["std"] / np.sqrt(v["n"]) if v else 0.0


# ── fig_ocr_3way (replaces fig_ocr_v2) ───────────────────────────────
def fig_ocr_3way(s, show_err=True):
    order = [m for m in METHODS if mean_of(s, m, "char_acc") is not None]
    means = [mean_of(s, m, "char_acc") for m in order]
    errs = [sem_of(s, m, "char_acc") for m in order] if show_err else None
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(range(len(order)), means, yerr=errs, capsize=3,
           color=[COL[m] for m in order], edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LBL[m] for m in order], fontsize=7)
    ax.set_ylabel("OCR character accuracy")
    ax.set_title("Downstream OCR @ matched 512px (n=55)", fontsize=8)
    top = max(means)
    for i, m in enumerate(means):
        off = (errs[i] if errs else 0) + 0.004
        ax.text(i, m + off, f"{m:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, top * 1.25)
    fig.tight_layout()
    fig.savefig("fig_ocr_3way.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_ocr_3way done")


# ── fig_refsr_tradeoff (NEW centrepiece) ─────────────────────────────
def fig_refsr_tradeoff(s):
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for m in METHODS:
        x, y = mean_of(s, m, "ssim"), mean_of(s, m, "char_acc")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=130, color=COL[m], edgecolor="black",
                   linewidth=0.8, zorder=3)
        ax.annotate(LBL[m].replace("\n", " "), (x, y), fontsize=7,
                    xytext=(6, 4), textcoords="offset points")
    bx, by = mean_of(s, "Input", "ssim"), mean_of(s, "Input", "char_acc")
    ax.axhline(by, color="#999", ls="--", lw=0.8, zorder=1)
    ax.axvline(bx, color="#999", ls="--", lw=0.8, zorder=1)
    ax.set_xlabel("SSIM  (fidelity \u2192)")
    ax.set_ylabel("OCR char. acc.  (readability \u2192)")
    ax.set_title("Fidelity vs. readability trade-off (512px)", fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_refsr_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_refsr_tradeoff done")


# ── fig_refsr_metrics (NEW 4-panel) ──────────────────────────────────
def fig_refsr_metrics(s):
    panels = [("ssim", "SSIM", "\u2191"), ("lpips", "LPIPS", "\u2193"),
              ("niqe", "NIQE", "\u2193"), ("char_acc", "OCR char", "\u2191")]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4))
    for ax, (k, lab_, arrow) in zip(axes, panels):
        means = [mean_of(s, m, k) for m in METHODS]
        ax.bar(range(len(METHODS)), means, color=[COL[m] for m in METHODS],
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LBL[m] for m in METHODS], fontsize=5.5)
        ax.set_title(f"{lab_} {arrow}", fontsize=8)
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.2f}" if v > 1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5)
    fig.suptitle("Matched-resolution comparison (512px, n=55): classical "
                 "fusion preserves fidelity; TTSR trades it for readability",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig("fig_refsr_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_refsr_metrics done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="compare_proc512.json")
    ap.add_argument("--errorbars", choices=["on", "off"], default="on")
    args = ap.parse_args()
    s = summ(load(args.results))
    print("Generating RefSR comparison figures...")
    fig_ocr_3way(s, show_err=(args.errorbars == "on"))
    fig_refsr_tradeoff(s)
    fig_refsr_metrics(s)
    print("Done.")


if __name__ == "__main__":
    main()
