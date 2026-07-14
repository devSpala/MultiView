"""
refsr_complete_eval.py — finish the RefSR comparison in ONE pass
================================================================
Your first RefSR run captured SSIM/PSNR/LapVar/runtime for TTSR but LPIPS and
NIQE came back empty (pyiqa not loaded) and OCR was not scored. This script
re-runs the registered RefSR adapter ONCE and captures everything:
  * full-reference SSIM, PSNR, LPIPS
  * no-reference NIQE
  * downstream OCR character/word accuracy vs human gt.txt
all on the same ROI and protocol as your main results.

PREREQUISITES (run these BEFORE importing this module, in the same session):
  !apt-get -qq install tesseract-ocr
  !pip -q install pytesseract jiwer pyiqa scipy
  import pyiqa            # <-- MUST be imported so NIQE/LPIPS are available
  import experimental_results_standalone   # your pipeline
  import refsr_baselines as rb
  from refsr_ttsr_adapter import make_ttsr_adapter
  rb.register("TTSR-rec", make_ttsr_adapter("TTSR","TTSR/TTSR.pt","cuda",4,"pm1"))

THEN:
  import refsr_complete_eval as rce
  rce.run("dataset", method="TTSR-rec")     # -> refsr_complete_TTSR-rec.json
"""

import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

import experimental_results_standalone as pipe
import refsr_baselines as rb

# reuse the corrected OCR helpers
import ocr_downstream_v2 as ocr


def run(dataset="dataset", method="TTSR-rec", max_dim=2000,
        out=None):
    out = out or f"refsr_complete_{method}.json"
    adapter = rb._ADAPTERS.get(method)
    if adapter is None:
        sys.exit(f"'{method}' not registered. Call rb.register(...) first.")

    root = Path(dataset)
    records = []
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        if gt is None:
            continue
        pair_dir = root / scenario / pair
        # proposed-method ROI so everything is scored on identical pixels
        ref_out = pipe.run_pipeline(m1, m2, pipe.config_for("P3_fidelity_lg"))
        roi = ref_out["roi"]
        gt_roi = pipe.crop(gt, roi)

        # human reference text (uses gt.txt you created)
        ref_text, src = ocr._reference_text(pair_dir, gt_roi)
        if len(ref_text) < ocr.MIN_REF_CHARS:
            ref_text = None  # OCR skipped but image metrics still recorded

        # run RefSR
        sr = adapter(m1, m2)
        if sr.shape[:2] != m1.shape[:2]:
            sr = cv2.resize(sr, (m1.shape[1], m1.shape[0]))
        sr_roi = pipe.crop(sr, roi)

        # full set of image metrics (pyiqa must be loaded for lpips/niqe)
        rec = pipe.score_roi(sr_roi, gt_roi)
        rec.update(scenario=scenario, pair=pair, method=method, aligned=True)

        # OCR
        if ref_text:
            ca, wa = ocr._accuracy(ref_text, ocr._ocr_text(sr))
            rec["char_acc"], rec["word_acc"] = ca, wa
        records.append(rec)
        print(f"  {scenario}/{pair} ssim={rec.get('ssim',0):.3f} "
              f"lpips={rec.get('lpips')} niqe={rec.get('niqe')} "
              f"char={rec.get('char_acc')}")

    # aggregate
    def agg(key):
        v = [r[key] for r in records if r.get(key) is not None]
        return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                    n=len(v)) if len(v) > 1 else None
    summary = {k: agg(k) for k in
               ["ssim","psnr","lpips","niqe","lapvar","runtime_ms",
                "char_acc","word_acc"]}
    json.dump({"records": records, "summary": summary, "method": method},
              open(out, "w"), indent=2)

    print(f"\n=== {method} (n={len(records)}) ===")
    for k in ["ssim","psnr","lpips","niqe","char_acc","word_acc"]:
        s = summary[k]
        print(f"  {k:10s} {s['mean']:.4f}" if s else f"  {k:10s} (missing)")
    print(f"Saved: {out}")
    return summary


if __name__ == "__main__":
    run()
