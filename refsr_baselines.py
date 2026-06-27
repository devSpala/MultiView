"""
refsr_baselines.py — compare against learned reference-based SR
================================================================
A top-tier reviewer will ask how the proposed method compares to learned
reference-based super-resolution (RefSR). This harness runs pretrained
RefSR models on the SAME pairs, with the SAME ROI and metrics as the main
experiments, so the comparison is fair. It also records each method's
wall-clock time, enabling the efficiency argument (the proposed classical
method is far cheaper and edge-deployable).

Two honest outcomes, both publishable:
  * The proposed method is competitive at a fraction of the compute
    (efficiency / edge claim), OR
  * RefSR wins on clean pairs but fails on the hard registration cases
    that learned matching rescues (robustness claim).

SUPPORTED BASELINES (pluggable)
-------------------------------
  C2-Matching   https://github.com/yumingj/C2-Matching
  TTSR          https://github.com/researchmm/TTSR
  SRNTT         https://github.com/ZZUTK/SRNTT  (TF; weights via repo)
  MASA-SR       https://github.com/dvlab-research/MASA-SR

Because each repo has its own loading API and heavy deps, this file does
NOT vendor them. Instead it defines a small ADAPTER protocol: you provide
a callable that maps (lr_bgr, ref_bgr) -> sr_bgr, register it, and the
harness handles dataset iteration, ROI cropping, metric computation, and
fair timing. Stub adapters with setup instructions are included.

USAGE
-----
  import refsr_baselines as rb
  # after installing a baseline repo and weights:
  rb.register("C2-Matching", make_c2matching_adapter("/path/to/weights"))
  rb.run_refsr("dataset", max_dim=2000)     # -> refsr_results.json
  rb.merge_into_main("results.json", "refsr_results.json")  # for figures
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

try:
    import experimental_results_standalone as pipe
except Exception as e:  # pragma: no cover
    pipe = None
    print(f"[warn] pipeline import failed: {e}", file=sys.stderr)


# ── adapter registry ─────────────────────────────────────────────────
# An adapter is: callable(lr_bgr: np.ndarray, ref_bgr: np.ndarray) -> sr_bgr
_ADAPTERS = {}


def register(name, fn):
    """Register a RefSR adapter under a display name."""
    _ADAPTERS[name] = fn
    print(f"[refsr] registered baseline: {name}")


def available():
    return list(_ADAPTERS)


# ── stub adapter factories (fill in per repo) ────────────────────────
def make_c2matching_adapter(weights_dir):
    """Returns an adapter for C2-Matching. Requires the C2-Matching repo
    importable and weights downloaded. Fill in the two marked lines with
    the repo's actual inference call; the surrounding glue is done."""
    def _adapter(lr_bgr, ref_bgr):
        # ---- BEGIN repo-specific (see C2-Matching README) -------------
        # from c2matching.inference import load_model, super_resolve
        # model = _adapter._model  # cache across calls
        # sr = super_resolve(model, lr_rgb, ref_rgb)   # returns RGB uint8
        # ---- END repo-specific ----------------------------------------
        raise NotImplementedError(
            "Fill in C2-Matching inference call. See the repo README at "
            "https://github.com/yumingj/C2-Matching ; weights dir: "
            f"{weights_dir}")
    return _adapter


def make_ttsr_adapter(weights_path):
    """Returns an adapter for TTSR (https://github.com/researchmm/TTSR)."""
    def _adapter(lr_bgr, ref_bgr):
        raise NotImplementedError(
            "Fill in TTSR inference call. See https://github.com/researchmm/TTSR ; "
            f"weights: {weights_path}")
    return _adapter


# ── evaluation ───────────────────────────────────────────────────────
def run_refsr(dataset="dataset", out="refsr_results.json", max_dim=2000):
    if pipe is None:
        sys.exit("pipeline module not importable.")
    if not _ADAPTERS:
        sys.exit("No RefSR adapters registered. Use register(name, fn) "
                 "after wiring up a baseline repo (see stub factories).")

    root = Path(dataset)
    records = []
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(root, max_dim):
        # Use the proposed method's ROI so RefSR is scored on identical
        # pixels as every other method in the main results.
        ref_out = pipe.run_pipeline(m1, m2, pipe.config_for("P3_fidelity_lg"))
        roi = ref_out["roi"]
        gt_roi = pipe.crop(gt, roi) if gt is not None else None

        for name, adapter in _ADAPTERS.items():
            t0 = time.perf_counter()
            try:
                sr = adapter(m1, m2)
                ok = True
            except NotImplementedError as e:
                print(f"[skip] {name}: {e}", file=sys.stderr)
                ok = False
            except Exception as e:  # robustness: log and continue
                print(f"[err] {name} on {scenario}/{pair}: {e}", file=sys.stderr)
                ok = False
            dt = (time.perf_counter() - t0) * 1000.0
            if not ok:
                continue
            if sr.shape[:2] != m1.shape[:2]:
                sr = cv2.resize(sr, (m1.shape[1], m1.shape[0]))
            rec = pipe.score_roi(pipe.crop(sr, roi), gt_roi)
            rec.update(scenario=scenario, pair=pair, method=name,
                       runtime_ms=dt, aligned=True)
            records.append(rec)
            print(f"  {scenario}/{pair} {name:<14s} "
                  f"ssim={rec['ssim'] if rec['ssim'] else float('nan'):.3f} "
                  f"lpips={rec['lpips'] if rec['lpips'] else float('nan'):.3f} "
                  f"t={dt:.0f}ms")

    if not records:
        sys.exit("No RefSR records produced (adapters all stubs?).")

    methods = sorted({r["method"] for r in records})
    summary = {}
    for m in methods:
        rows = [r for r in records if r["method"] == m]
        def agg(key):
            v = [r[key] for r in rows if r.get(key) is not None]
            return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                        n=len(v)) if v else None
        summary[m] = {k: agg(k) for k in
                      ("ssim", "psnr", "lpips", "niqe", "lapvar", "runtime_ms")}

    Path(out).write_text(json.dumps(
        dict(records=records, summary=summary), indent=2))
    print("\n=== RefSR baselines (proposed ROI, same metrics) ===")
    print(f"{'Method':<16s}{'SSIM':>8s}{'LPIPS':>8s}{'NIQE':>8s}{'ms':>9s}")
    for m in methods:
        s = summary[m]
        def f(k, p=3):
            return "-" if s[k] is None else f"{s[k]['mean']:.{p}f}"
        print(f"{m:<16s}{f('ssim'):>8s}{f('lpips'):>8s}{f('niqe'):>8s}"
              f"{f('runtime_ms',0):>9s}")
    print(f"Saved: {out}")
    return summary


def merge_into_main(main_json="results.json", refsr_json="refsr_results.json",
                    out=None):
    """Append RefSR records into the main results.json so the existing
    figure scripts plot them alongside the proposed method."""
    main = json.load(open(main_json))
    refsr = json.load(open(refsr_json))
    main["records"].extend(refsr["records"])
    main.setdefault("summary", {}).update(refsr["summary"])
    out = out or main_json
    Path(out).write_text(json.dumps(main, indent=2))
    print(f"Merged {len(refsr['records'])} RefSR records into {out}")


if __name__ == "__main__":
    print("RefSR baseline harness. Register adapters then call run_refsr().")
    print("Registered:", available() or "(none yet)")
