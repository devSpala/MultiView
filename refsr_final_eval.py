"""
refsr_final_eval.py — single deterministic RefSR evaluation
===========================================================
Fixes the run1-vs-run2 discrepancy and the missing LPIPS/NIQE/runtime by:
  * scoring image metrics on a COPY of the SR image, BEFORE any OCR
    preprocessing touches it (prevents in-place mutation differences);
  * timing the adapter call explicitly (captures runtime);
  * requiring pyiqa to be importable and FAILING LOUDLY if LPIPS/NIQE
    cannot be computed, instead of silently writing None;
  * saving every SR image to  refsr_sr_out/<scenario>_<pair>.png  so the
    numbers are auditable and reproducible.

PREREQUISITES (same session, BEFORE importing this file):
  !apt-get -qq install tesseract-ocr
  !pip -q install pytesseract jiwer pyiqa scipy
  import pyiqa; print("pyiqa OK", pyiqa.__version__)   # MUST succeed
  import experimental_results_standalone, refsr_baselines
  from refsr_ttsr_adapter import make_ttsr_adapter
  refsr_baselines.register("TTSR-rec",
      make_ttsr_adapter("TTSR","TTSR/TTSR.pt","cuda",4,"pm1"))

RUN:
  import refsr_final_eval as rfe
  rfe.run("dataset", method="TTSR-rec")
"""

import json, sys, time, os
from pathlib import Path
import numpy as np
import cv2

import experimental_results_standalone as pipe
import refsr_baselines as rb
import ocr_downstream_v2 as ocr

# Hard requirement: pyiqa must be present so LPIPS/NIQE are real, not None.
try:
    import pyiqa  # noqa
    _HAVE_PYIQA = True
except Exception:
    _HAVE_PYIQA = False


def run(dataset="dataset", method="TTSR-rec", max_dim=2000, out=None,
        save_dir="refsr_sr_out"):
    if not _HAVE_PYIQA:
        sys.exit("pyiqa not importable -> LPIPS/NIQE would be None. "
                 "Run:  !pip install pyiqa   then  import pyiqa  first.")
    adapter = rb._ADAPTERS.get(method)
    if adapter is None:
        sys.exit(f"'{method}' not registered.")
    out = out or f"refsr_final_{method}.json"
    Path(save_dir).mkdir(exist_ok=True)

    root = Path(dataset)
    records = []
    n_missing_metric = 0
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        pair_dir = root / scenario / pair
        ref_out = pipe.run_pipeline(m1, m2, pipe.config_for("P3_fidelity_lg"))
        roi = ref_out["roi"]
        gt_roi = pipe.crop(gt, roi)

        # --- run RefSR with explicit timing ---
        t0 = time.perf_counter()
        sr = adapter(m1, m2)
        dt = (time.perf_counter() - t0) * 1000.0
        if sr.shape[:2] != m1.shape[:2]:
            sr = cv2.resize(sr, (m1.shape[1], m1.shape[0]))

        # save SR image for auditability
        cv2.imwrite(os.path.join(save_dir, f"{scenario}_{pair}.png"), sr)

        # --- score image metrics on a COPY, before OCR touches anything ---
        sr_roi = pipe.crop(sr.copy(), roi).copy()
        rec = pipe.score_roi(sr_roi, gt_roi.copy())
        rec.update(scenario=scenario, pair=pair, method=method,
                   aligned=True, runtime_ms=dt)
        if rec.get("lpips") is None or rec.get("niqe") is None:
            n_missing_metric += 1

        # --- OCR on a separate copy, against human gt.txt ---
        ref_text, src = ocr._reference_text(pair_dir, gt_roi.copy())
        if ref_text and len(ref_text) >= ocr.MIN_REF_CHARS:
            ca, wa = ocr._accuracy(ref_text, ocr._ocr_text(sr.copy()))
            rec["char_acc"], rec["word_acc"] = ca, wa
            rec["gt_source"] = src
        records.append(rec)
        print(f"  {scenario}/{pair} ssim={rec.get('ssim',0):.3f} "
              f"lpips={rec.get('lpips')} niqe={rec.get('niqe')} "
              f"char={rec.get('char_acc')} t={dt:.0f}ms")

    if n_missing_metric:
        print(f"\n[WARN] {n_missing_metric} pairs had None LPIPS/NIQE despite "
              "pyiqa being loaded -- check score_roi's metric init.")

    def agg(key):
        v = [r[key] for r in records if r.get(key) is not None]
        return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                    n=len(v)) if len(v) > 1 else None
    summary = {k: agg(k) for k in
               ["ssim","psnr","lpips","niqe","lapvar","runtime_ms",
                "char_acc","word_acc"]}
    json.dump({"records": records, "summary": summary, "method": method,
               "gt_source": "human (gt.txt)"}, open(out, "w"), indent=2)

    print(f"\n=== {method} (n={len(records)}) ===")
    for k in ["ssim","psnr","lpips","niqe","char_acc","word_acc","runtime_ms"]:
        s = summary[k]
        print(f"  {k:11s} {s['mean']:.4f}" if s else f"  {k:11s} MISSING")
    print(f"SR images saved to: {save_dir}/   |   Saved: {out}")
    return summary


if __name__ == "__main__":
    run()
