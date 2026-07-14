"""
ocr_downstream_v2.py — corrected downstream OCR evaluation
===========================================================
Fixes four logic flaws found in v1 that made the measurement unreliable:

  FIX 1  Real ground-truth text. Prefer a human-typed  gt.txt  per pair.
         Only fall back to OCR-of-clean-image, and when doing so, DISCARD
         pairs whose clean image is itself unreadable (an unreadable
         reference makes CER/WER meaningless).

  FIX 2  OCR the full enhanced image (optionally upscaled), not the tiny
         common-view ROI crop, which starved Tesseract of pixels.

  FIX 3  OCR-oriented preprocessing: grayscale -> upscale -> Otsu
         binarization -> pad, with --psm 6 (uniform text block).

  FIX 4  Report only on pairs with a USABLE reference (>= MIN_REF_CHARS
         readable characters), and report that count explicitly.

USAGE (Colab)
-------------
  !apt-get -qq install tesseract-ocr
  !pip -q install pytesseract jiwer
  import ocr_downstream_v2 as ocr
  ocr.run_ocr("dataset", max_dim=2000)     # -> ocr_results_v2.json
  ocr.make_ocr_figure()                     # -> fig_ocr_v2.png

STRONGLY RECOMMENDED: add gt.txt files (see make_gt_txt_stubs()).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

try:
    import experimental_results_standalone as pipe
except Exception as e:  # pragma: no cover
    pipe = None
    print(f"[warn] pipeline import failed: {e}", file=sys.stderr)
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False
try:
    from jiwer import cer, wer
    HAVE_JIWER = True
except Exception:
    HAVE_JIWER = False


PROPOSED = "P3_fidelity_lg"
DEFAULT_METHODS = ["Base_M1", "V5_full", PROPOSED]
NICE = {"Base_M1": "Degraded input", "B2_homog_alpha": "Homography + blend",
        "V5_full": "Aggressive sharpening", "P1_fidelity": "Fusion (no post)",
        "P3_fidelity_lg": "Proposed"}

MIN_REF_CHARS = 10      # a pair needs at least this many reference chars
OCR_UPSCALE = 2.0       # upscale factor before OCR (helps small text)
TESS_CONFIG = "--psm 6" # assume a uniform block of text


# ── OCR with proper preprocessing (FIX 3) ────────────────────────────
def _prep_for_ocr(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if OCR_UPSCALE and OCR_UPSCALE != 1.0:
        g = cv2.resize(g, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                       interpolation=cv2.INTER_CUBIC)
    # Otsu binarization (auto threshold); invert if background is dark
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() < 127:           # mostly dark -> invert so text is dark-on-light
        b = 255 - b
    b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return b


def _ocr_text(bgr):
    img = _prep_for_ocr(bgr)
    txt = pytesseract.image_to_string(img, config=TESS_CONFIG)
    return " ".join(txt.split()).strip().lower()


def _accuracy(ref, hyp):
    if not ref:
        return (np.nan, np.nan)
    if not hyp:
        return (0.0, 0.0)
    c = max(0.0, min(1.0, 1.0 - float(cer(ref, hyp))))
    w = max(0.0, min(1.0, 1.0 - float(wer(ref, hyp))))
    return (c, w)


# ── reference text (FIX 1) ───────────────────────────────────────────
def _reference_text(pair_dir, gt_full_bgr):
    """Human gt.txt if present (preferred); else OCR of the FULL clean
    image. Returns (text, source). Empty text -> caller discards pair."""
    txtp = pair_dir / "gt.txt"
    if txtp.exists():
        t = " ".join(txtp.read_text(encoding="utf-8").split()).strip().lower()
        return t, "human"
    return _ocr_text(gt_full_bgr), "ocr_gt"


# ── main ─────────────────────────────────────────────────────────────
def run_ocr(dataset="dataset", out="ocr_results_v2.json", max_dim=2000,
            methods=None):
    if not (HAVE_TESS and HAVE_JIWER):
        sys.exit("Install: apt-get install tesseract-ocr && pip install pytesseract jiwer")
    if pipe is None:
        sys.exit("pipeline module not importable.")
    methods = methods or DEFAULT_METHODS
    root = Path(dataset)
    records = []
    src_counts = defaultdict(int)
    discarded_unreadable = 0

    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        pair_dir = root / scenario / pair
        # FIX 2: OCR the FULL images (no ROI crop starving Tesseract)
        ref_text, src = _reference_text(pair_dir, gt)
        # FIX 4: require a usable reference
        if len(ref_text) < MIN_REF_CHARS:
            discarded_unreadable += 1
            print(f"[discard] {scenario}/{pair}: reference unreadable "
                  f"({len(ref_text)} chars) — excluded from OCR stats",
                  file=sys.stderr)
            continue
        src_counts[src] += 1

        # produce each method's FULL enhanced image and OCR it
        ref_out = pipe.run_pipeline(m1, m2, pipe.config_for(PROPOSED))
        for method in methods:
            if method == "Base_M1":
                img, aligned = m1, True
            else:
                o = (ref_out if method == PROPOSED
                     else pipe.run_pipeline(m1, m2, pipe.config_for(method)))
                img, aligned = o["result"], o["aligned"]
            ca, wa = _accuracy(ref_text, _ocr_text(img))
            records.append(dict(scenario=scenario, pair=pair, method=method,
                                char_acc=ca, word_acc=wa, aligned=aligned,
                                ref_len=len(ref_text)))
            print(f"  {scenario}/{pair} {method:<16s} "
                  f"char={ca:.3f} word={wa:.3f}")

    if not records:
        sys.exit("No usable OCR pairs. Add gt.txt files — your clean images "
                 "are not reliably OCR-readable on their own.")

    summary = {}
    for m in methods:
        rows = [r for r in records if r["method"] == m
                and (m == "Base_M1" or r["aligned"])]
        ca = [r["char_acc"] for r in rows if not np.isnan(r["char_acc"])]
        wa = [r["word_acc"] for r in rows if not np.isnan(r["word_acc"])]
        summary[m] = dict(
            char_acc=dict(mean=float(np.mean(ca)), std=float(np.std(ca, ddof=1)),
                          n=len(ca)) if len(ca) > 1 else None,
            word_acc=dict(mean=float(np.mean(wa)), std=float(np.std(wa, ddof=1)),
                          n=len(wa)) if len(wa) > 1 else None)

    # paired uplift proposed vs input
    uplift = {}
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except Exception:
        have_scipy = False
    idx = defaultdict(dict)
    for r in records:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    paired = [k for k in idx if "Base_M1" in idx[k] and PROPOSED in idx[k]
              and idx[k][PROPOSED]["aligned"]]
    for metric in ("char_acc", "word_acc"):
        b = [idx[k]["Base_M1"][metric] for k in paired]
        p = [idx[k][PROPOSED][metric] for k in paired]
        d = dict(input_mean=float(np.mean(b)), proposed_mean=float(np.mean(p)),
                 abs_gain=float(np.mean(p) - np.mean(b)), n=len(paired),
                 pairs_improved=int(sum(1 for x, y in zip(b, p) if y > x)))
        if have_scipy and len(paired) >= 6 and any(x != y for x, y in zip(b, p)):
            try:
                _, pv = wilcoxon(p, b); d["p_value"] = float(pv)
            except ValueError:
                pass
        uplift[metric] = d

    payload = dict(records=records, summary=summary, uplift=uplift,
                   methods=methods, gt_source=dict(src_counts),
                   usable_pairs=len(paired),
                   discarded_unreadable=discarded_unreadable,
                   config=dict(min_ref_chars=MIN_REF_CHARS,
                               ocr_upscale=OCR_UPSCALE, tess=TESS_CONFIG))
    Path(out).write_text(json.dumps(payload, indent=2))

    print(f"\n=== OCR accuracy (usable pairs: {len(paired)}; "
          f"discarded unreadable: {discarded_unreadable}) ===")
    print(f"{'Method':<22s}{'CharAcc':>10s}{'WordAcc':>10s}")
    for m in methods:
        s = summary[m]
        ca = f"{s['char_acc']['mean']:.3f}" if s['char_acc'] else "-"
        wa = f"{s['word_acc']['mean']:.3f}" if s['word_acc'] else "-"
        print(f"{NICE.get(m, m):<22s}{ca:>10s}{wa:>10s}")
    print("\nProposed vs degraded input (paired):")
    for metric, d in uplift.items():
        line = (f"  {metric}: {d['input_mean']:.3f} -> {d['proposed_mean']:.3f} "
                f"(gain {d['abs_gain']:+.3f}), improved {d['pairs_improved']}/{d['n']}")
        if "p_value" in d:
            line += f", p={d['p_value']:.4g}"
        print(line)
    if src_counts.get("ocr_gt"):
        print(f"\n[warn] {src_counts['ocr_gt']} pairs used OCR-of-clean-image as "
              "reference. This is unreliable (your clean crops barely OCR). "
              "Add gt.txt files for a valid result — see make_gt_txt_stubs().")
    print(f"Saved: {out}")
    return payload


# ── helper to bootstrap gt.txt files ─────────────────────────────────
def make_gt_txt_stubs(dataset="dataset", max_dim=2000):
    """Write a gt_suggested.txt next to each pair containing the OCR of the
    clean image, so you can quickly correct it by hand into gt.txt. Saves
    typing — you just fix Tesseract's mistakes rather than transcribe from
    scratch."""
    if not HAVE_TESS:
        sys.exit("tesseract not available.")
    root = Path(dataset); n = 0
    seen = set()
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        # one transcription per SCENE (pairs share a scene's text); key on pair index
        pd = root / scenario / pair
        guess = _ocr_text(gt)
        (pd / "gt_suggested.txt").write_text(guess, encoding="utf-8")
        n += 1
    print(f"Wrote {n} gt_suggested.txt files. Review each, correct the text, "
          "and save as gt.txt in the same folder. Then rerun run_ocr().")


# ── figure ───────────────────────────────────────────────────────────
def make_ocr_figure(results="ocr_results_v2.json", out="fig_ocr_v2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 8, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    d = json.load(open(results))
    methods = d["methods"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(methods)); w = 0.38
    for j, metric in enumerate(("char_acc", "word_acc")):
        vals = [d["summary"][m][metric]["mean"] if d["summary"][m][metric] else 0
                for m in methods]
        ax.bar(x + (j - 0.5) * w, vals, w,
               color=("#4C72B0" if j == 0 else "#2E8B57"), alpha=0.9, zorder=3,
               label=("Character accuracy" if j == 0 else "Word accuracy"))
        for i, v in enumerate(vals):
            ax.text(i + (j - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center",
                    fontsize=6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(m, m) for m in methods], fontsize=6.5,
                       rotation=15, ha="right")
    ax.set_ylabel("OCR accuracy (vs. reference text)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.92)
    ax.set_title(f"Downstream OCR ({d.get('usable_pairs','?')} usable pairs)",
                 fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="ocr_results_v2.json")
    ap.add_argument("--max_dim", type=int, default=2000)
    ap.add_argument("--make_stubs", action="store_true",
                    help="write gt_suggested.txt files to bootstrap gt.txt")
    args = ap.parse_args()
    if args.make_stubs:
        make_gt_txt_stubs(args.dataset, args.max_dim)
    else:
        run_ocr(args.dataset, args.out, args.max_dim)


if __name__ == "__main__":
    main()
