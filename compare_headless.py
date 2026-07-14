"""
compare_headless.py — fair paper-numbers harness
=================================================
Runs, at a SINGLE matched resolution for every method:
  * Input (degraded base)
  * Proposed (improved core: LightGlue + multiscale pyramid + guided weight)
  * Proposed (ablations: single-scale, no-guided) for the paper's ablation
  * TTSR-rec (if a TTSR adapter is registered)
and scores SSIM/PSNR/LPIPS/NIQE + OCR char/word against human gt.txt.

Design rules baked in (so the numbers are reviewer-safe):
  - EVERY method is cropped to the same ROI and resized to the same
    `proc` size before scoring. No method gets a resolution advantage.
  - Metrics are computed on copies, before OCR preprocessing touches
    anything (avoids in-place mutation differences).
  - pyiqa must be importable or LPIPS/NIQE fail loudly (no silent None).

USAGE (Colab), after importing pyiqa FIRST then the pipeline:
  import compare_headless as ch
  ch.run("dataset", proc=512,
         ttsr_adapter=make_ttsr_adapter(...),   # optional
         gt_txt=True)
"""

import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

import enhance_core as ec

# pipeline metric/ocr helpers (your existing modules)
try:
    import experimental_results_standalone as pipe
    _HAVE_PIPE = True
except Exception:
    _HAVE_PIPE = False
try:
    import ocr_downstream_v2 as ocr
    _HAVE_OCR = True
except Exception:
    _HAVE_OCR = False


def _resize_long(img, proc):
    h, w = img.shape[:2]
    s = proc / max(h, w)
    return cv2.resize(img, (int(round(w*s)), int(round(h*s))))


def run(dataset="dataset", proc=512, max_dim=2000, ttsr_adapter=None,
        gt_txt=True, out=None, device="cpu"):
    if not _HAVE_PIPE:
        sys.exit("experimental_results_standalone not importable.")
    if not (pipe.HAVE_LPIPS and pipe.HAVE_NIQE):
        sys.exit("LPIPS/NIQE unavailable: import pyiqa BEFORE the pipeline.")
    out = out or f"compare_proc{proc}.json"
    root = Path(dataset)

    # methods: name -> callable(m1, m2) -> full-frame BGR result
    def _proposed(m1, m2, multiscale, guided):
        r = ec.enhance(m1, m2, matcher="auto", multiscale=multiscale,
                       guided=guided, use_post=False, device=device)
        return r["result"], r["aligned"]
    methods = {
        "Input":             lambda a, b: (a, True),
        "Proposed":          lambda a, b: _proposed(a, b, True, True),
        "Proposed (single)": lambda a, b: _proposed(a, b, False, False),
    }
    if ttsr_adapter is not None:
        methods["TTSR-rec"] = lambda a, b: (ttsr_adapter(a, b), True)

    records = []
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        pair_dir = root / scenario / pair
        # ROI from the proposed alignment so all methods share pixels
        r = ec.enhance(m1, m2, matcher="auto", device=device)
        roi = r["roi"] or (0, 0, m1.shape[1], m1.shape[0])
        gt_roi = _resize_long(pipe.crop(gt, roi), proc)

        ref_text = None
        if gt_txt and _HAVE_OCR:
            rt, _ = ocr._reference_text(pair_dir, gt_roi.copy())
            if rt and len(rt) >= ocr.MIN_REF_CHARS:
                ref_text = rt

        for name, fn in methods.items():
            t0 = time.perf_counter()
            full, aligned = fn(m1, m2)
            dt = (time.perf_counter() - t0) * 1000.0
            if full.shape[:2] != m1.shape[:2]:
                full = cv2.resize(full, (m1.shape[1], m1.shape[0]))
            img = _resize_long(pipe.crop(full, roi), proc)
            rec = pipe.score_roi(img.copy(), gt_roi.copy())
            rec.update(scenario=scenario, pair=pair, method=name,
                       aligned=aligned, runtime_ms=dt)
            if ref_text and _HAVE_OCR:
                ca, wa = ocr._accuracy(ref_text, ocr._ocr_text(img.copy()))
                rec["char_acc"], rec["word_acc"] = ca, wa
            records.append(rec)
        print(f"  {scenario}/{pair} done")

    # aggregate + paired tests
    names = list(methods)
    summary = {}
    for m in names:
        rows = [r for r in records if r["method"] == m]
        def agg(k):
            v = [r[k] for r in rows if r.get(k) is not None]
            return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                        n=len(v)) if len(v) > 1 else None
        summary[m] = {k: agg(k) for k in
                      ["ssim","psnr","lpips","niqe","char_acc","word_acc",
                       "runtime_ms"]}

    from scipy.stats import wilcoxon
    idx = defaultdict(dict)
    for r in records:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    tests = {}
    pairs_ocr = [k for k in idx
                 if all(m in idx[k] and "char_acc" in idx[k][m] for m in names)]
    base = "Input"
    for m in names:
        if m == base:
            continue
        a = [idx[k][m]["char_acc"] for k in pairs_ocr]
        b = [idx[k][base]["char_acc"] for k in pairs_ocr]
        if a and any(x != y for x, y in zip(a, b)):
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = None
            tests[f"{m}_vs_Input"] = dict(
                mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                a_better=int(sum(1 for x, y in zip(a, b) if x > y)),
                n=len(pairs_ocr), p_value=p)

    json.dump({"records": records, "summary": summary, "tests": tests,
               "proc": proc}, open(out, "w"), indent=2)

    print(f"\n=== Matched comparison @ {proc}px ===")
    print(f"{'Method':20s}{'SSIM':>8s}{'PSNR':>8s}{'LPIPS':>8s}{'NIQE':>8s}{'OCRc':>8s}{'ms':>8s}")
    for m in names:
        s = summary[m]
        def f(k, p=3):
            return "--" if not s[k] else f"{s[k]['mean']:.{p}f}"
        print(f"{m:20s}{f('ssim'):>8s}{f('psnr',2):>8s}{f('lpips'):>8s}"
              f"{f('niqe',2):>8s}{f('char_acc'):>8s}{f('runtime_ms',0):>8s}")
    print("\nPaired OCR char-accuracy vs Input:")
    for n, t in tests.items():
        pv = f"p={t['p_value']:.2e}" if t['p_value'] is not None else "p=n/a"
        print(f"  {n:24s} {t['mean_a']:.3f} vs {t['mean_b']:.3f} "
              f"({t['a_better']}/{t['n']}, {pv})")
    print(f"\nSaved: {out}")
    return summary


if __name__ == "__main__":
    run()
