"""
matched_resolution_eval.py — fair same-resolution comparison
============================================================
Option 1: measure INPUT, PROPOSED, and TTSR all at ONE fixed internal
resolution, so the OCR / fidelity comparison is fair and reproducible.

Why: TTSR cannot process full-res frames (OOM), so the only honest
head-to-head is to evaluate every method at the same reduced resolution.
This script does exactly that in a single pass and captures every metric
(SSIM, PSNR, LPIPS, NIQE, OCR char/word) plus runtime.

It also FIXES the LPIPS/NIQE = None bug: those depend on pyiqa being
imported BEFORE experimental_results_standalone. This script checks that
and tells you to fix the import order if needed.

------------------------------------------------------------------
PREREQUISITES (run in this exact order, in a fresh session):

  !apt-get -qq install tesseract-ocr
  !pip -q install pytesseract jiwer pyiqa scipy

  import pyiqa                              # (1) pyiqa FIRST
  print("pyiqa", pyiqa.__version__)

  import experimental_results_standalone as pipe   # (2) pipeline SECOND
  print("LPIPS available:", pipe.HAVE_LPIPS, "| NIQE available:", pipe.HAVE_NIQE)
  # ^ both MUST print True. If False, restart runtime and redo in order.

  import refsr_baselines
  from refsr_ttsr_adapter import make_ttsr_adapter
  refsr_baselines.register("TTSR-rec",
      make_ttsr_adapter("TTSR","TTSR/TTSR.pt","cuda",4,"pm1",max_proc=384))

THEN:
  import matched_resolution_eval as mre
  mre.run("dataset", proc=384)      # every method evaluated at 384px
------------------------------------------------------------------
"""

import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

import experimental_results_standalone as pipe
import refsr_baselines as rb
import ocr_downstream_v2 as ocr


def _resize_long(img, proc):
    """Resize so the long side == proc, preserving aspect ratio."""
    h, w = img.shape[:2]
    s = proc / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))))


def run(dataset="dataset", proc=384, max_dim=2000, out=None):
    if not (pipe.HAVE_LPIPS and pipe.HAVE_NIQE):
        sys.exit("LPIPS/NIQE unavailable: import pyiqa BEFORE "
                 "experimental_results_standalone, then restart and retry.")
    out = out or f"matched_eval_proc{proc}.json"
    ttsr = rb._ADAPTERS.get("TTSR-rec")
    if ttsr is None:
        sys.exit("TTSR-rec not registered; register the adapter first.")

    root = Path(dataset)
    records = []
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        pair_dir = root / scenario / pair

        # produce full-res outputs first
        prop_out = pipe.run_pipeline(m1, m2, pipe.config_for("P3_fidelity_lg"))
        proposed_full = prop_out["result"]
        roi = prop_out["roi"]

        t0 = time.perf_counter()
        ttsr_full = ttsr(m1, m2)
        ttsr_ms = (time.perf_counter() - t0) * 1000.0
        if ttsr_full.shape[:2] != m1.shape[:2]:
            ttsr_full = cv2.resize(ttsr_full, (m1.shape[1], m1.shape[0]))

        # ROI-crop each method, THEN resize every crop to the SAME proc size
        gt_roi   = _resize_long(pipe.crop(gt, roi), proc)
        variants = {
            "Input":    _resize_long(pipe.crop(m1, roi), proc),
            "Proposed": _resize_long(pipe.crop(proposed_full, roi), proc),
            "TTSR-rec": _resize_long(pipe.crop(ttsr_full, roi), proc),
        }

        # human reference text (gt.txt) — identical for all methods
        ref_text, src = ocr._reference_text(pair_dir, gt_roi.copy())
        ocr_ok = ref_text and len(ref_text) >= ocr.MIN_REF_CHARS

        for name, img in variants.items():
            rec = pipe.score_roi(img.copy(), gt_roi.copy())
            rec.update(scenario=scenario, pair=pair, method=name)
            if name == "TTSR-rec":
                rec["runtime_ms"] = ttsr_ms
            if ocr_ok:
                ca, wa = ocr._accuracy(ref_text, ocr._ocr_text(img.copy()))
                rec["char_acc"], rec["word_acc"] = ca, wa
            records.append(rec)
        print(f"  {scenario}/{pair}  "
              + "  ".join(f"{n}:ssim={[r for r in records if r['scenario']==scenario and r['pair']==pair and r['method']==n][0]['ssim']:.2f}"
                          for n in variants))

    # aggregate per method
    methods = ["Input", "Proposed", "TTSR-rec"]
    summary = {}
    for m in methods:
        rows = [r for r in records if r["method"] == m]
        def agg(key):
            v = [r[key] for r in rows if r.get(key) is not None]
            return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                        n=len(v)) if len(v) > 1 else None
        summary[m] = {k: agg(k) for k in
                      ["ssim","psnr","lpips","niqe","lapvar","runtime_ms",
                       "char_acc","word_acc"]}

    # paired significance: proposed vs input, proposed vs TTSR (char_acc)
    from scipy.stats import wilcoxon
    idx = defaultdict(dict)
    for r in records:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    tests = {}
    keys = [k for k in idx if all(m in idx[k] for m in methods)
            and all("char_acc" in idx[k][m] for m in methods)]
    for a, b in [("Proposed","Input"), ("Proposed","TTSR-rec"),
                 ("TTSR-rec","Input")]:
        va = [idx[k][a]["char_acc"] for k in keys]
        vb = [idx[k][b]["char_acc"] for k in keys]
        try:
            _, p = wilcoxon(va, vb)
        except ValueError:
            p = None
        tests[f"{a}_vs_{b}"] = dict(
            mean_a=float(np.mean(va)), mean_b=float(np.mean(vb)),
            a_better=int(sum(1 for x, y in zip(va, vb) if x > y)),
            n=len(keys), p_value=p)

    json.dump({"records": records, "summary": summary, "tests": tests,
               "proc": proc, "gt_source": "human (gt.txt)"},
              open(out, "w"), indent=2)

    print(f"\n=== Matched-resolution comparison @ {proc}px (n pairs) ===")
    print(f"{'Method':12s}{'SSIM':>8s}{'PSNR':>8s}{'LPIPS':>8s}{'NIQE':>8s}{'OCRc':>8s}{'OCRw':>8s}")
    for m in methods:
        s = summary[m]
        def f(k, p=3):
            return "--" if not s[k] else f"{s[k]['mean']:.{p}f}"
        print(f"{m:12s}{f('ssim'):>8s}{f('psnr',2):>8s}{f('lpips'):>8s}"
              f"{f('niqe',2):>8s}{f('char_acc'):>8s}{f('word_acc'):>8s}")
    print("\nPaired OCR char-accuracy tests:")
    for name, t in tests.items():
        pv = f"p={t['p_value']:.2e}" if t['p_value'] is not None else "p=n/a"
        print(f"  {name:22s} {t['mean_a']:.3f} vs {t['mean_b']:.3f}  "
              f"({t['a_better']}/{t['n']} better, {pv})")
    print(f"\nSaved: {out}")
    return summary


if __name__ == "__main__":
    run()
