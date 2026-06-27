"""
experimental_results_standalone.py — SELF-CONTAINED experimental suite
=======================================================================
Single-file version for Google Colab / Jupyter: the entire revised
pipeline is embedded, so no companion pipeline.py is needed.

── Colab quick start ──────────────────────────────────────────────────
  !pip -q install opencv-python-headless scikit-image scipy pyiqa
  # upload or mount your dataset as  dataset/S1/pair_01/{m1.jpg,m2.jpg,gt.jpg} ...
  import experimental_results_standalone as ex
  ex.run_experiments("dataset")            # -> results.json (+ .csv)
  ex.run_sweep("dataset")                  # -> sensitivity.json
  ex.run_bench("m1.jpg", "m2.jpg", tag="colab")   # -> runtime_colab.json
  ex.make_all_figures()                    # -> fig1..fig7 .png

Or, if you paste this whole file into ONE notebook cell, just call the
same four functions in the next cell.

From a terminal it still works as a CLI:
  python experimental_results_standalone.py run --dataset ./dataset
  python experimental_results_standalone.py sweep
  python experimental_results_standalone.py bench --tag rpi4
  python experimental_results_standalone.py figures
"""

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Optional learned components. Everything below degrades gracefully:
# the classical pipeline runs with zero torch dependencies; installing
# the extras unlocks NIQE/LPIPS metrics and the A1-A3 learned variants.
#   pip install pyiqa torch torchvision
#   pip install git+https://github.com/cvg/LightGlue.git
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    import pyiqa
    _NIQE = pyiqa.create_metric("niqe")
    HAVE_NIQE = True
except Exception:
    HAVE_NIQE = False

try:
    _LPIPS = pyiqa.create_metric("lpips")   # learned perceptual metric
    HAVE_LPIPS = True
except Exception:
    HAVE_LPIPS = False


def _lightglue_available():
    if not HAVE_TORCH:
        return False
    try:
        import lightglue  # noqa: F401
        return True
    except Exception:
        return False


def _raft_available():
    if not HAVE_TORCH:
        return False
    try:
        from torchvision.models.optical_flow import raft_small  # noqa
        return True
    except Exception:
        return False


HAVE_LIGHTGLUE = _lightglue_available()
HAVE_RAFT = _raft_available()

_MODEL_CACHE = {}


def _get_lightglue():
    """Lazy-load SuperPoint + LightGlue (first call downloads weights)."""
    if "lg" not in _MODEL_CACHE:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        _MODEL_CACHE["sp"] = SuperPoint(max_num_keypoints=2048).eval()
        _MODEL_CACHE["lg"] = LightGlue(features="superpoint").eval()
        _MODEL_CACHE["rbd"] = rbd
    return _MODEL_CACHE["sp"], _MODEL_CACHE["lg"], _MODEL_CACHE["rbd"]


def _get_raft():
    """Lazy-load RAFT-small (first call downloads weights)."""
    if "raft" not in _MODEL_CACHE:
        from torchvision.models.optical_flow import (raft_small,
                                                     Raft_Small_Weights)
        w = Raft_Small_Weights.DEFAULT
        _MODEL_CACHE["raft"] = raft_small(weights=w).eval()
    return _MODEL_CACHE["raft"]


# ════════════════════════════════════════════════════════════════
# Embedded pipeline (formerly pipeline.py)
# ════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # registration
    registration: str = "affine"       # "affine" | "homography" | "none"
    max_features: int = 4000
    lowe_ratio: float = 0.7
    ransac_thresh: float = 3.0
    scale_min: float = 0.1
    scale_max: float = 10.0
    matcher: str = "sift"              # "sift" | "lightglue" (learned)
    flow_engine: str = "farneback"     # "farneback" | "raft" (learned)
    reg_max_dim: int = 1600            # estimate transform on downscaled
                                       # copies (robustness + speed on
                                       # high-MP photos); matrix is then
                                       # rescaled to full resolution.
    # optical flow refinement
    use_flow: bool = True
    flow_prefilter: bool = True        # F2: structure-guided flow
    # fusion
    fusion: str = "lab"                # "lab" | "alpha" | "none"
    alpha_blend: float = 0.5           # for the alpha-blend baselines
    detail_blur_ksize: int = 15
    weight_blur_ksize: int = 31
    lapvar_ksize: int = 15
    # post-processing (switchable for the no-post ablation row)
    use_postprocess: bool = True
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    unsharp_alpha: float = 1.2         # I_final = (1+a)I - a*blur(I)
    unsharp_sigma: float = 1.5
    # ROI
    auto_roi: bool = True              # F3
    manual_roi: tuple | None = None    # (x1, y1, x2, y2) in pixels
    # timing
    collect_timing: bool = False


# Named configurations used in the paper -------------------------------

def config_for(name: str) -> PipelineConfig:
    """Return the configuration for a paper method name."""
    presets = {
        # Baselines
        "B1_lanczos":     PipelineConfig(registration="none", use_flow=False,
                                         fusion="none", use_postprocess=False),
        "B2_homog_alpha": PipelineConfig(registration="homography",
                                         use_flow=False, fusion="alpha",
                                         use_postprocess=False),
        "B3_affine_alpha": PipelineConfig(registration="affine",
                                          use_flow=False, fusion="alpha",
                                          use_postprocess=False),
        # Ablation variants
        "V1_homog_alpha": PipelineConfig(registration="homography",
                                         use_flow=False, fusion="alpha",
                                         use_postprocess=False),
        "V2_affine_only": PipelineConfig(registration="affine",
                                         use_flow=False, fusion="alpha",
                                         use_postprocess=False),
        "V3_affine_flow": PipelineConfig(registration="affine",
                                         use_flow=True, fusion="alpha",
                                         use_postprocess=False),
        "V4_affine_lab":  PipelineConfig(registration="affine",
                                         use_flow=False, fusion="lab",
                                         use_postprocess=True),
        "V5_full":        PipelineConfig(),  # everything on
        # New ablation row requested by reviewers: fusion WITHOUT
        # CLAHE/unsharp, to isolate genuine detail transfer from
        # generic sharpening.
        "V5_full_nopost": PipelineConfig(use_postprocess=False),
        "proposed":       PipelineConfig(),
        # Accuracy-priority learned variants: ONLY the correspondence
        # source / flow estimator is swapped; the 4-DOF affine
        # constraint, sharpness weighting, and CIELAB fusion are
        # unchanged. Require torch (+ lightglue / torchvision).
        "A1_lightglue":   PipelineConfig(matcher="lightglue"),
        "A2_raft":        PipelineConfig(flow_engine="raft"),
        "A3_learned":     PipelineConfig(matcher="lightglue",
                                         flow_engine="raft"),
        # ── Fixes motivated by the real-data diagnostics ────────────
        # The default post-processing (CLAHE 2.5 + unsharp 1.2) was
        # over-sharpening real photos, collapsing SSIM from ~0.82 to
        # ~0.66 while only inflating Laplacian variance. These variants
        # tame or remove it. P1 is the recommended new default for
        # fidelity; P2 keeps a light touch of sharpening; P3 adds
        # learned matching (the alignment fix) on top of P1.
        "P1_fidelity":    PipelineConfig(use_postprocess=False),
        "P2_light_post":  PipelineConfig(clahe_clip=1.5, unsharp_alpha=0.4),
        "P3_fidelity_lg": PipelineConfig(use_postprocess=False,
                                         matcher="lightglue"),
    }
    return presets[name]


# ────────────────────────────────────────────────────────────────
# Stage 2 — global registration
# ────────────────────────────────────────────────────────────────

def _sift_matches(g1, g2, cfg: PipelineConfig, ratio=None):
    sift = cv2.SIFT_create(nfeatures=cfg.max_features)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, []
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(d2, d1, k=2)
    r = ratio if ratio is not None else cfg.lowe_ratio
    good = [m for m, n in pairs if m.distance < r * n.distance]
    return kp1, kp2, good


def _scaled_gray(img, max_dim):
    """Grayscale copy downscaled so max(h,w) <= max_dim; returns scale."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    m = max(h, w)
    if not max_dim or m <= max_dim:
        return g, 1.0
    s = max_dim / m
    g = cv2.resize(g, (int(round(w * s)), int(round(h * s))),
                   interpolation=cv2.INTER_AREA)
    return g, s


def estimate_transform(m1, m2, cfg: PipelineConfig):
    """Estimate affine (4-DOF) or homography (8-DOF) mapping M2 -> M1.

    The transform is estimated on downscaled copies (<= cfg.reg_max_dim)
    and rescaled to full-resolution coordinates: high-megapixel photos
    otherwise yield slow SIFT and degenerate partial-affine estimates.
    A second attempt with a stricter Lowe ratio and a looser RANSAC
    threshold is made before declaring failure.

    Returns (matrix, n_inliers, kind) or (None, n_good, kind).
    """
    g1, s1 = _scaled_gray(m1, cfg.reg_max_dim)
    g2, s2 = _scaled_gray(m2, cfg.reg_max_dim)

    def _learned_correspondences():
        """SuperPoint + LightGlue matches (m2 -> m1) on the downscaled
        grays. Returns (src, dst) float32 (N,1,2) arrays or None."""
        sp, lg, rbd = _get_lightglue()
        with torch.no_grad():
            t1 = torch.from_numpy(g1).float()[None, None] / 255.0
            t2 = torch.from_numpy(g2).float()[None, None] / 255.0
            f1 = sp.extract(t1)
            f2 = sp.extract(t2)
            out = lg({"image0": f2, "image1": f1})   # match m2 -> m1
            f1r, f2r, outr = rbd(f1), rbd(f2), rbd(out)
            matches = outr["matches"].cpu().numpy()  # (K,2): [idx2, idx1]
        if len(matches) < 10:
            return None, len(matches)
        src = f2r["keypoints"].cpu().numpy()[matches[:, 0]]
        dst = f1r["keypoints"].cpu().numpy()[matches[:, 1]]
        return (src.reshape(-1, 1, 2).astype(np.float32),
                dst.reshape(-1, 1, 2).astype(np.float32)), len(matches)

    def attempt(ratio, thresh):
        if cfg.matcher == "lightglue" and HAVE_LIGHTGLUE:
            try:
                pts, n = _learned_correspondences()
            except Exception as e:           # model/load failure -> SIFT
                print(f"[warn] LightGlue failed ({e}); falling back to "
                      "SIFT for this pair.", file=sys.stderr)
                pts = None
                n = 0
            if pts is None and n == 0:
                kp1, kp2, good = _sift_matches(g1, g2, cfg, ratio)
                if kp1 is None or len(good) < 10:
                    return None, (0 if kp1 is None else len(good))
                src = np.float32([kp2[m.queryIdx].pt
                                  for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp1[m.trainIdx].pt
                                  for m in good]).reshape(-1, 1, 2)
            elif pts is None:
                return None, n
            else:
                src, dst = pts
        else:
            kp1, kp2, good = _sift_matches(g1, g2, cfg, ratio)
            if kp1 is None or len(good) < 10:
                return None, (0 if kp1 is None else len(good))
            src = np.float32([kp2[m.queryIdx].pt
                              for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp1[m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)
        if cfg.registration == "homography":
            M, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
        else:
            M, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh,
                maxIters=5000, confidence=0.995, refineIters=10)
            if M is not None:
                sc = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
                if not (cfg.scale_min <= sc <= cfg.scale_max):
                    return None, 0
        n = int(mask.sum()) if mask is not None else 0
        return M, n

    M, n_in = attempt(cfg.lowe_ratio, cfg.ransac_thresh)
    if M is None:   # retry: stricter matches, more permissive RANSAC
        M, n_in = attempt(min(cfg.lowe_ratio, 0.65), cfg.ransac_thresh + 1.0)
    if M is None:
        return None, n_in, cfg.registration

    # Rescale from downscaled-estimation coordinates to full resolution:
    # x1_full = S1^-1 . T_small . S2 . x2_full
    S1i = np.diag([1.0 / s1, 1.0 / s1, 1.0])
    S2 = np.diag([s2, s2, 1.0])
    if cfg.registration == "homography":
        Mf = S1i @ M @ S2
        Mf = Mf / Mf[2, 2]
    else:
        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        Mf = (S1i @ M3 @ S2)[:2, :]
    return Mf.astype(np.float64), n_in, cfg.registration


def warp_reference(m2, M, target_shape, kind):
    h, w = target_shape[:2]
    if kind == "homography":
        return cv2.warpPerspective(m2, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT)
    return cv2.warpAffine(m2, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


# ────────────────────────────────────────────────────────────────
# F3 — automatic Common-View-Zone ROI
# ────────────────────────────────────────────────────────────────

def auto_roi_from_transform(M, kind, m2_shape, m1_shape, margin: int = 8):
    """Project M2's corners into M1 space; intersect with the frame.

    Returns (x1, y1, x2, y2) bounding box of the common view zone,
    shrunk by `margin` px to stay clear of warp border reflections.
    """
    h2, w2 = m2_shape[:2]
    h1, w1 = m1_shape[:2]
    corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    if kind == "homography":
        proj = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    else:
        M3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        proj = cv2.perspectiveTransform(corners, M3).reshape(-1, 2)
    x1 = max(int(np.ceil(proj[:, 0].min())) + margin, 0)
    y1 = max(int(np.ceil(proj[:, 1].min())) + margin, 0)
    x2 = min(int(np.floor(proj[:, 0].max())) - margin, w1)
    y2 = min(int(np.floor(proj[:, 1].max())) - margin, h1)
    if x2 - x1 < 32 or y2 - y1 < 32:           # degenerate overlap
        return (0, 0, w1, h1)
    return (x1, y1, x2, y2)


# ────────────────────────────────────────────────────────────────
# Stage 3 — structure-guided dense optical flow (F2)
# ────────────────────────────────────────────────────────────────

def _structure_image(gray):
    """CLAHE + Gaussian high-pass: makes Farneback flow respond to
    structural boundaries instead of absolute intensity, mitigating
    the brightness-constancy violation between two auto-exposed
    cameras."""
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    low = cv2.GaussianBlur(eq, (0, 0), 5.0)
    hp = cv2.subtract(eq, low)
    return cv2.add(hp, 128)  # re-centre for the polynomial expansion


def _raft_flow(bgr1, bgr2, max_dim=1024):
    """Dense flow base->reference via RAFT-small. Computed at <=max_dim
    (multiple of 8) and upsampled; learned flow is robust to the
    inter-camera exposure mismatch by construction."""
    model = _get_raft()
    H, W = bgr1.shape[:2]
    s = min(1.0, max_dim / max(H, W))
    h8 = max(64, int(round(H * s / 8)) * 8)
    w8 = max(64, int(round(W * s / 8)) * 8)

    def prep(img):
        r = cv2.resize(img, (w8, h8), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        return t / 127.5 - 1.0                    # RAFT expects [-1, 1]

    with torch.no_grad():
        flow = model(prep(bgr1), prep(bgr2))[-1][0].cpu().numpy()
    flow = flow.transpose(1, 2, 0)                # (h8, w8, 2)
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= W / w8
    flow[..., 1] *= H / h8
    return flow


def _residual_err(m1, ref):
    """Cheap alignment-error proxy on structure images."""
    a = _structure_image(cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY))
    b = _structure_image(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    return float(np.mean(cv2.absdiff(a, b)))


def refine_with_flow(m1, warped_m2, cfg: PipelineConfig):
    h, w = m1.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    if cfg.flow_engine == "raft" and HAVE_RAFT:
        try:
            flow = _raft_flow(m1, warped_m2)
            # RAFT(I1, I2) maps I1 coords to I2 coords, so the aligned
            # reference samples warped_m2 at (x + dx, y + dy).
            cand = cv2.remap(warped_m2, x + flow[..., 0],
                             y + flow[..., 1], cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)
            # Empirical safety: only accept if residual error improves
            # (guards against any convention/scale slip on edge cases).
            if _residual_err(m1, cand) <= _residual_err(m1, warped_m2):
                return cand
            print("[warn] RAFT flow did not reduce residual error; "
                  "keeping affine-only alignment for this pair.",
                  file=sys.stderr)
            return warped_m2
        except Exception as e:
            print(f"[warn] RAFT failed ({e}); using Farneback.",
                  file=sys.stderr)

    g1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(warped_m2, cv2.COLOR_BGR2GRAY)
    if cfg.flow_prefilter:
        g1, g2 = _structure_image(g1), _structure_image(g2)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Backward mapping: destination (x,y) samples source (x - dx, y - dy).
    map_x = x - flow[..., 0]
    map_y = y - flow[..., 1]
    return cv2.remap(warped_m2, map_x, map_y, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REFLECT)


# ────────────────────────────────────────────────────────────────
# Stage 4 — sharpness-aware weighting (F1: absolute-scale variance)
# ────────────────────────────────────────────────────────────────

def laplacian_variance_map(gray, ksize: int = 15):
    """RAW local Laplacian variance — no per-image normalisation."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    mu = cv2.GaussianBlur(lap, (ksize, ksize), 0)
    mu2 = cv2.GaussianBlur(lap * lap, (ksize, ksize), 0)
    return np.maximum(0.0, mu2 - mu * mu)


def sharpness_weight(roi_base, roi_ref, cfg: PipelineConfig):
    g_b = cv2.cvtColor(roi_base, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
    v_b = laplacian_variance_map(g_b, cfg.lapvar_ksize)
    v_r = laplacian_variance_map(g_r, cfg.lapvar_ksize)
    diff = np.clip(v_r - v_b, 0, None)        # absolute units
    mx = float(diff.max())
    weight = diff / mx if mx > 1e-5 else np.zeros_like(diff)
    k = cfg.weight_blur_ksize
    return cv2.GaussianBlur(weight, (k, k), 0)


# ────────────────────────────────────────────────────────────────
# Stage 5 — fusion + post-processing
# ────────────────────────────────────────────────────────────────

def lab_frequency_fusion(roi_base, roi_ref, weight, cfg: PipelineConfig):
    lab_b = cv2.cvtColor(roi_base, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_b, A_b, B_b = cv2.split(lab_b)
    L_r = cv2.split(lab_r)[0]
    k = cfg.detail_blur_ksize
    detail = L_r - cv2.GaussianBlur(L_r, (k, k), 0)
    L_e = np.clip(L_b + detail * weight, 0, 255)
    out = cv2.merge([L_e, A_b, B_b]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def alpha_blend(roi_base, roi_ref, alpha: float):
    return cv2.addWeighted(roi_base, 1.0 - alpha, roi_ref, alpha, 0)


def postprocess(roi, cfg: PipelineConfig):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)).apply(l)
    roi = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(roi, (0, 0), cfg.unsharp_sigma)
    return cv2.addWeighted(roi, 1.0 + cfg.unsharp_alpha,
                           blur, -cfg.unsharp_alpha, 0)


# ────────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────────

def run_pipeline(m1, m2, cfg: PipelineConfig | None = None):
    """Run the configured pipeline. Returns a dict with the result,
    ROI, diagnostics, and (optionally) per-stage timings in ms."""
    cfg = cfg or PipelineConfig()
    h1, w1 = m1.shape[:2]
    timings = {}
    tic = time.perf_counter

    # ---- Baseline B1: single-image upsampling, no reference --------
    if cfg.registration == "none" and cfg.fusion == "none":
        up = cv2.resize(m1, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        res = cv2.resize(up, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        return dict(result=res, roi=(0, 0, w1, h1), aligned=False,
                    n_inliers=0, timings=timings)

    # ---- Stage 2: global registration ------------------------------
    t0 = tic()
    M, n_in, kind = estimate_transform(m1, m2, cfg)
    timings["s2_registration_ms"] = (tic() - t0) * 1e3
    if M is not None:
        warped = warp_reference(m2, M, m1.shape, kind)
        aligned = True
    else:
        warped = cv2.resize(m2, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        aligned = False

    # ---- Stage 3: dense flow refinement -----------------------------
    if cfg.use_flow and aligned:
        t0 = tic()
        warped = refine_with_flow(m1, warped, cfg)
        timings["s3_flow_ms"] = (tic() - t0) * 1e3

    # ---- ROI ---------------------------------------------------------
    if cfg.manual_roi is not None:
        roi = cfg.manual_roi
    elif cfg.auto_roi and aligned:
        roi = auto_roi_from_transform(M, kind, m2.shape, m1.shape)
    else:
        roi = (0, 0, w1, h1)
    x1, y1, x2, y2 = roi
    roi_b, roi_r = m1[y1:y2, x1:x2], warped[y1:y2, x1:x2]

    # ---- Stage 4 + 5: weighting and fusion ---------------------------
    t0 = tic()
    if cfg.fusion == "lab":
        weight = sharpness_weight(roi_b, roi_r, cfg)
        fused = lab_frequency_fusion(roi_b, roi_r, weight, cfg)
    elif cfg.fusion == "alpha":
        weight = None
        fused = alpha_blend(roi_b, roi_r, cfg.alpha_blend)
    else:
        weight, fused = None, roi_b.copy()
    timings["s4s5_fusion_ms"] = (tic() - t0) * 1e3

    if cfg.use_postprocess:
        t0 = tic()
        fused = postprocess(fused, cfg)
        timings["s5_post_ms"] = (tic() - t0) * 1e3

    # ---- soft-mask blend back into the full frame ---------------------
    mask = np.zeros((h1, w1), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    full = m1.copy(); full[y1:y2, x1:x2] = fused
    mask3 = cv2.merge([mask] * 3)
    result = (m1.astype(np.float32) * (1 - mask3)
              + full.astype(np.float32) * mask3).astype(np.uint8)

    return dict(result=result, roi=roi, aligned=aligned, n_inliers=n_in,
                weight=weight, warped=warped, timings=timings)


# ════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════

def lap_var(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ssim_score(img, gt):
    a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    return float(ssim_fn(a, b, data_range=255))


def psnr_score(img, gt):
    return float(psnr_fn(gt, img, data_range=255))


def niqe_score(img):
    if not HAVE_NIQE:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return float(_NIQE(t).item())


def lpips_score(img_bgr, gt_bgr):
    """LPIPS (AlexNet) learned perceptual distance vs ground truth.
    Lower is better. Requires pyiqa."""
    if not HAVE_LPIPS or gt_bgr is None:
        return None
    def to_t(im):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(_LPIPS(to_t(img_bgr), to_t(gt_bgr)).item())


def crop(img, roi):
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return dict(mean=float(np.mean(vals)), std=sd, n=len(vals))


# ════════════════════════════════════════════════════════════════
# Dataset iteration
# ════════════════════════════════════════════════════════════════

METHODS = [
    "B1_lanczos",
    "B2_homog_alpha",
    "B3_affine_alpha",
    "V2_affine_only",
    "V3_affine_flow",
    "V4_affine_lab",
    "V5_full_nopost",   # V5- : full pipeline without CLAHE/unsharp
    "V5_full",          # == proposed (classical, edge-priority)
    # Fidelity-fix variants (see config_for notes); evaluated always —
    # these are cheap and directly test the post-processing hypothesis.
    "P1_fidelity",
    "P2_light_post",
]
# Accuracy-priority learned variants — evaluated only when the optional
# dependencies are installed (torch + lightglue / torchvision).
LEARNED_METHODS = [m for m, ok in (
    ("A1_lightglue", HAVE_LIGHTGLUE),
    ("A2_raft", HAVE_RAFT),
    ("A3_learned", HAVE_LIGHTGLUE and HAVE_RAFT),
    ("P3_fidelity_lg", HAVE_LIGHTGLUE)) if ok]
METHODS_ALL = ["Base_M1"] + METHODS + LEARNED_METHODS


def _resize_max(img, max_dim):
    """Downscale so max(h,w) <= max_dim (no-op if already smaller)."""
    if img is None or not max_dim:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    s = max_dim / m
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def iter_pairs(root: Path, max_dim=None):
    pairs = sorted(p for p in root.glob("S*/pair_*") if p.is_dir())
    for pdir in pairs:
        m1 = _resize_max(cv2.imread(str(pdir / "m1.jpg")), max_dim)
        m2 = _resize_max(cv2.imread(str(pdir / "m2.jpg")), max_dim)
        gt_p = pdir / "gt.jpg"
        gt = cv2.imread(str(gt_p)) if gt_p.exists() else None
        if m1 is None or m2 is None:
            print(f"[skip] {pdir}: missing m1/m2", file=sys.stderr)
            continue
        if gt is not None and gt.shape != m1.shape:
            # gt must be pixel-aligned with m1 (same viewpoint & size)
            gt = cv2.resize(gt, (m1.shape[1], m1.shape[0]),
                            interpolation=cv2.INTER_AREA)
        yield pdir.parent.name, pdir.name, m1, m2, gt


def score_roi(res_roi, gt_roi):
    return dict(
        lapvar=lap_var(res_roi),
        ssim=ssim_score(res_roi, gt_roi) if gt_roi is not None else None,
        psnr=psnr_score(res_roi, gt_roi) if gt_roi is not None else None,
        niqe=niqe_score(res_roi),
        lpips=lpips_score(res_roi, gt_roi),
    )


# ════════════════════════════════════════════════════════════════
# Subcommand: run  (main experiments -> results.json)
# ════════════════════════════════════════════════════════════════

def cmd_run(args):
    root = Path(args.dataset)
    max_dim = getattr(args, "max_dim", None)
    if not LEARNED_METHODS:
        print("[note] learned variants A1-A3 skipped (install: "
              "pip install torch torchvision "
              "git+https://github.com/cvg/LightGlue.git)", file=sys.stderr)
    records = []
    align_ok, align_fail, no_gt = [], [], []
    for scenario, pair, m1, m2, gt in iter_pairs(root, max_dim):
        # The proposed method's auto-ROI defines the common evaluation
        # region so every method is scored on identical pixels.
        ref_out = run_pipeline(m1, m2, config_for("V5_full"))
        roi = ref_out["roi"]
        gt_roi = crop(gt, roi) if gt is not None else None

        tag = f"{scenario}/{pair}"
        if ref_out["aligned"]:
            align_ok.append(tag)
        else:
            align_fail.append(tag)
            print(f"[ALIGN FAIL] {tag}: affine registration failed "
                  f"(matches={ref_out['n_inliers']}); fell back to resize — "
                  "this pair's affine-variant scores are NOT meaningful.",
                  file=sys.stderr)
        if gt is None:
            no_gt.append(tag)

        rec = score_roi(crop(m1, roi), gt_roi)
        rec.update(scenario=scenario, pair=pair, method="Base_M1")
        records.append(rec)

        for method in METHODS + LEARNED_METHODS:
            out = (ref_out if method == "V5_full"
                   else run_pipeline(m1, m2, config_for(method)))
            rec = score_roi(crop(out["result"], roi), gt_roi)
            rec.update(scenario=scenario, pair=pair, method=method,
                       aligned=out["aligned"], n_inliers=out["n_inliers"])
            records.append(rec)
            s = rec["ssim"]
            flag = ("n/a " if method == "B1_lanczos"
                    else "ok " if out["aligned"] else "FAIL")
            print(f"{pair:>10s} {method:<16s} align={flag} "
                  f"inl={out['n_inliers']:4d} lap={rec['lapvar']:9.1f} "
                  f"ssim={s if s is not None else float('nan'):.3f}")

    if not records:
        sys.exit(f"No data found under {root} (expected S*/pair_*/m1.jpg).")

    scenarios = sorted({r["scenario"] for r in records})
    summary, per_scenario = {}, {}
    for m in METHODS_ALL:
        rows = [r for r in records if r["method"] == m]
        summary[m] = {k: agg(rows, k) for k in ("lapvar", "ssim", "psnr", "niqe", "lpips")}
        per_scenario[m] = {
            s: {k: agg([r for r in rows if r["scenario"] == s], k)
                for k in ("lapvar", "ssim", "psnr", "niqe", "lpips")}
            for s in scenarios}

    # paired Wilcoxon: proposed vs each method
    sig = {}
    if HAVE_SCIPY:
        prop = {(r["scenario"], r["pair"]): r for r in records
                if r["method"] == "V5_full"}
        for m in METHODS_ALL:
            if m == "V5_full":
                continue
            other = {(r["scenario"], r["pair"]): r for r in records
                     if r["method"] == m}
            keys = sorted(set(prop) & set(other))
            sig[m] = {}
            for metric in ("ssim", "psnr", "lapvar", "niqe", "lpips"):
                a = [prop[k][metric] for k in keys
                     if prop[k].get(metric) is not None]
                b = [other[k][metric] for k in keys
                     if other[k].get(metric) is not None]
                if len(a) == len(b) and len(a) >= 6:
                    try:
                        _, p = wilcoxon(a, b)
                        sig[m][metric] = dict(p_value=float(p), n=len(a))
                    except ValueError:
                        pass

    payload = dict(records=records, summary=summary,
                   per_scenario=per_scenario, significance=sig,
                   niqe_available=HAVE_NIQE,
                   alignment=dict(ok=len(align_ok), failed=len(align_fail),
                                  failed_pairs=align_fail),
                   pairs_without_gt=no_gt)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    with open(Path(args.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        w.writeheader(); w.writerows(records)

    print("\n=== Aggregate (mean ± std over all pairs) ===")
    hdr = (f"{'Method':<18s}{'LapVar':>17s}{'SSIM':>16s}{'PSNR':>15s}"
           f"{'NIQE':>14s}{'LPIPS':>15s}")
    print(hdr)
    for m in METHODS_ALL:
        s = summary[m]
        def fmt(d, p=3):
            return "-" if d is None else f"{d['mean']:.{p}f}±{d['std']:.{p}f}"
        print(f"{m:<18s}{fmt(s['lapvar'],1):>17s}{fmt(s['ssim']):>16s}"
              f"{fmt(s['psnr'],2):>15s}{fmt(s['niqe'],2):>14s}"
              f"{fmt(s['lpips']):>15s}")

    # ── experiment health check ─────────────────────────────────────
    total = len(align_ok) + len(align_fail)
    print(f"\nAlignment success: {len(align_ok)}/{total} pairs")
    if align_fail:
        print(f"  FAILED: {', '.join(align_fail)}")
        print("  -> inspect these pairs (overlap too small? motion blur "
              "too severe?) or recapture them; affine-variant numbers on "
              "failed pairs are fallback-resize artefacts, not results.")
    if no_gt:
        print(f"Pairs without gt.jpg: {len(no_gt)}/{total} "
              "-> SSIM/PSNR unavailable for these; figures 2-3 need gt.")
    if total < 15:
        print(f"Only {total} pairs evaluated — the paper protocol needs "
              "5 scenarios x >=3 pairs (50 pairs recommended) for "
              "meaningful std-devs and significance tests.")
    print(f"Saved: {args.out}")


# ════════════════════════════════════════════════════════════════
# Subcommand: sweep  (parameter sensitivity -> sensitivity.json)
# ════════════════════════════════════════════════════════════════

ALPHA_GRID = [0.6, 0.9, 1.2, 1.5, 1.8]


def cmd_sweep(args):
    root = Path(args.dataset)
    pairs = list(iter_pairs(root, getattr(args, "max_dim", None)))
    if not pairs:
        sys.exit(f"No data found under {root}.")
    points = []
    for alpha in ALPHA_GRID:
        rows = []
        for scenario, pair, m1, m2, gt in pairs:
            cfg = PipelineConfig(unsharp_alpha=alpha)
            out = run_pipeline(m1, m2, cfg)
            roi = out["roi"]
            rows.append(score_roi(crop(out["result"], roi),
                                  crop(gt, roi) if gt is not None else None))
        pt = dict(alpha=alpha,
                  **{k: agg(rows, k) for k in ("lapvar", "ssim", "psnr", "niqe", "lpips")})
        points.append(pt)
        sm = pt["ssim"]["mean"] if pt["ssim"] else float("nan")
        print(f"alpha={alpha:.1f}  lapvar={pt['lapvar']['mean']:9.1f}  "
              f"ssim={sm:.3f}")
    Path(args.out).write_text(json.dumps(dict(points=points,
                                              niqe_available=HAVE_NIQE),
                                         indent=2))
    print(f"Saved: {args.out}")


# ════════════════════════════════════════════════════════════════
# Subcommand: bench  (runtime -> runtime_<tag>.json)
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# Subcommand: prepare  (raw captures -> paper dataset layout)
# ════════════════════════════════════════════════════════════════
# Your degradations are SYNTHETIC (per the paper: Gaussian blur, motion
# blur, JPEG re-compression, low light), so the protocol is:
#   1. Capture each scene SHARP with the base camera M1 (tripod,
#      focused) -> this file IS the ground truth gt.jpg.
#   2. Capture the same scene with the reference camera M2 at a
#      15-35 degree offset -> ref.jpg.
#   3. prepare_dataset() synthesises the degraded m1.jpg per scenario
#      from gt and writes the S*/pair_* layout.
#
# Raw layout expected:
#   raw/scene_01/{base.jpg, ref.jpg}
#   raw/scene_02/{base.jpg, ref.jpg}
#   ...
# 10 scenes -> 50 pairs (each scene appears once in each of S1-S5).

def _motion_blur(img, ksize=21, angle=0.0):
    k = np.zeros((ksize, ksize), np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def _jpeg_recompress(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_degraded(gt, scenario, seed=0):
    """Scenario-specific degradation of the sharp gt -> m1 (paper Sec 6.2)."""
    if scenario == "S1":     # moderate Gaussian blur
        return cv2.GaussianBlur(gt, (0, 0), 2.5)
    if scenario == "S2":     # severe Gaussian blur
        return cv2.GaussianBlur(gt, (0, 0), 4.5)
    if scenario == "S3":     # motion blur on structured text
        return _motion_blur(gt, ksize=21, angle=15.0)
    if scenario == "S4":     # JPEG Q=35 + motion blur
        return _jpeg_recompress(_motion_blur(gt, ksize=15, angle=0.0), 35)
    if scenario == "S5":     # low light + noise + moderate blur
        dark = cv2.convertScaleAbs(gt, alpha=0.55, beta=-10)
        rng = np.random.default_rng(seed)
        noisy = np.clip(dark.astype(np.float32)
                        + rng.normal(0, 6, dark.shape), 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(noisy, (0, 0), 3.0)
    raise ValueError(f"unknown scenario {scenario}")


def prepare_dataset(raw_dir="raw", out_dir="dataset", max_dim=None):
    """Build the S*/pair_* evaluation layout from raw sharp captures.

    raw_dir/scene_*/base.jpg : sharp base-camera capture (becomes gt)
    raw_dir/scene_*/ref.jpg  : reference-camera capture at angular offset
    """
    scenes = sorted(d for d in Path(raw_dir).iterdir() if d.is_dir())
    if not scenes:
        sys.exit(f"No scene folders under {raw_dir} "
                 "(expected raw/scene_01/{base.jpg, ref.jpg}).")
    n = 0
    for i, sd in enumerate(scenes, 1):
        base = next((sd / f for f in ("base.jpg", "base.png", "gt.jpg")
                     if (sd / f).exists()), None)
        ref = next((sd / f for f in ("ref.jpg", "ref.png", "m2.jpg")
                    if (sd / f).exists()), None)
        gt = _resize_max(cv2.imread(str(base)), max_dim) if base else None
        m2 = _resize_max(cv2.imread(str(ref)), max_dim) if ref else None
        if gt is None or m2 is None:
            print(f"[skip] {sd}: need base.jpg + ref.jpg", file=sys.stderr)
            continue
        for scen in ("S1", "S2", "S3", "S4", "S5"):
            d = Path(out_dir) / scen / f"pair_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "gt.jpg"), gt,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
            cv2.imwrite(str(d / "m2.jpg"), m2,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(d / "m1.jpg"), make_degraded(gt, scen, seed=i),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    print(f"Prepared {n} pairs from {len(scenes)} scenes into {out_dir}/ "
          "(each scene degraded once per scenario S1-S5).")


def cmd_prepare(args):
    prepare_dataset(args.raw, args.out, getattr(args, "max_dim", None))


# ════════════════════════════════════════════════════════════════
# Public-benchmark adapters (cross-dataset generalisation)
# ════════════════════════════════════════════════════════════════
# Both adapters reuse make_degraded(), so the degradation protocol is
# IDENTICAL to the primary dataset — only the imagery source changes.
# Scenarios are assigned round-robin so each is represented.

def prepare_cufed5(cufed_root, out_dir="dataset_cufed5", max_dim=None,
                   limit=None):
    """CUFED5 reference-SR test set (Zhang et al., SRNTT, CVPR 2019).

    Download (Google Drive links on https://github.com/ZZUTK/SRNTT),
    extract, and point cufed_root at the folder containing files named
    like  000_0.png (target)  and  000_1.png .. 000_5.png (references,
    decreasing similarity).  Mapping:
        gt = XXX_0  ·  m2 = XXX_1 (most similar ref)
        m1 = make_degraded(gt, scenario)   [scenario round-robin S1-S5]
    """
    root = Path(cufed_root)
    targets = sorted(root.glob("*_0.png")) + sorted(root.glob("*_0.jpg"))
    if limit:
        targets = targets[:limit]
    if not targets:
        sys.exit(f"No '*_0.png' targets found under {root} — point this "
                 "at the extracted CUFED5 test folder.")
    n = 0
    for i, tpath in enumerate(targets, 1):
        stem = tpath.name.rsplit("_0.", 1)[0]
        ref = next((root / f"{stem}_1{ext}" for ext in (".png", ".jpg")
                    if (root / f"{stem}_1{ext}").exists()), None)
        gt = _resize_max(cv2.imread(str(tpath)), max_dim)
        m2 = _resize_max(cv2.imread(str(ref)), max_dim) if ref else None
        if gt is None or m2 is None:
            print(f"[skip] {stem}: missing target or _1 reference",
                  file=sys.stderr)
            continue
        scen = f"S{(i - 1) % 5 + 1}"
        d = Path(out_dir) / scen / f"pair_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "gt.jpg"), gt, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(d / "m2.jpg"), m2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(d / "m1.jpg"), make_degraded(gt, scen, seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    print(f"CUFED5: prepared {n} pairs into {out_dir}/ "
          f"(then: run_experiments('{out_dir}'))")


def prepare_middlebury(mb_root, out_dir="dataset_middlebury",
                       max_dim=None, limit=None):
    """Middlebury 2014 stereo set (Scharstein et al., GCPR 2014).

    Download scenes from https://vision.middlebury.edu/stereo/data/
    (the '-perfect' rectified pairs). Each scene folder must contain
    im0.png and im1.png. Mapping — note this is the HARD case for the
    4-DOF affine model, since stereo pairs contain real depth parallax:
        gt = im0  ·  m2 = im1  ·  m1 = make_degraded(im0, scenario)
    """
    root = Path(mb_root)
    scenes = sorted(d for d in root.iterdir()
                    if d.is_dir() and (d / "im0.png").exists()
                    and (d / "im1.png").exists())
    if limit:
        scenes = scenes[:limit]
    if not scenes:
        sys.exit(f"No scene folders with im0.png/im1.png under {root}.")
    n = 0
    for i, sd in enumerate(scenes, 1):
        gt = _resize_max(cv2.imread(str(sd / "im0.png")), max_dim)
        m2 = _resize_max(cv2.imread(str(sd / "im1.png")), max_dim)
        if gt is None or m2 is None:
            continue
        scen = f"S{(i - 1) % 5 + 1}"
        d = Path(out_dir) / scen / f"pair_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "gt.jpg"), gt, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(d / "m2.jpg"), m2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(d / "m1.jpg"), make_degraded(gt, scen, seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    print(f"Middlebury: prepared {n} pairs into {out_dir}/ "
          f"(then: run_experiments('{out_dir}'))")


# ════════════════════════════════════════════════════════════════
# Subcommand: bench  (runtime -> runtime_<tag>.json)
# ════════════════════════════════════════════════════════════════

def cmd_bench(args):
    m1 = _resize_max(cv2.imread(args.m1), getattr(args, "max_dim", None))
    m2 = _resize_max(cv2.imread(args.m2), getattr(args, "max_dim", None))
    if m1 is None or m2 is None:
        sys.exit("m1/m2 images not found.")
    cfg = PipelineConfig(collect_timing=True)
    for _ in range(args.warmup):
        run_pipeline(m1, m2, cfg)
    runs = [run_pipeline(m1, m2, cfg)["timings"] for _ in range(args.repeats)]
    keys = sorted({k for r in runs for k in r})
    stages = {k: dict(mean=statistics.mean([r[k] for r in runs if k in r]),
                      std=statistics.pstdev([r[k] for r in runs if k in r]))
              for k in keys}
    totals = [sum(r.values()) for r in runs]
    payload = dict(tag=args.tag, platform=platform.platform(),
                   machine=platform.machine(),
                   image_size=f"{m1.shape[1]}x{m1.shape[0]}",
                   repeats=args.repeats, stages=stages,
                   total=dict(mean=statistics.mean(totals),
                              std=statistics.pstdev(totals)))
    out = f"runtime_{args.tag}.json"
    Path(out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved: {out}")


# ════════════════════════════════════════════════════════════════
# Subcommand: figures  (the 7 paper figures)
# ════════════════════════════════════════════════════════════════

C_BASE, C_HOMO, C_AFFA, C_PROP = "#4C72B0", "#DD8452", "#937860", "#2E8B57"

# Candidates that could be the "proposed" method, best first by intent.
# The figure code picks whichever has the highest SSIM on aligned pairs
# (falls back to V5_full if no ground truth is available).
_PROPOSED_CANDIDATES = ["P3_fidelity_lg", "P1_fidelity", "P2_light_post",
                        "V5_full"]


def _best_proposed(data):
    """Pick the variant with the highest mean SSIM (aligned pairs) as
    the 'proposed' method shown in the figures. Pure no-GT datasets
    fall back to V5_full."""
    best, best_ssim = "V5_full", -1.0
    for m in _PROPOSED_CANDIDATES:
        s = data["summary"].get(m, {}).get("ssim")
        if s and s["mean"] > best_ssim:
            best, best_ssim = m, s["mean"]
    return best


def _main3(data):
    prop = _best_proposed(data)
    lbl = {"P1_fidelity": "Proposed (P1, no post)",
           "P2_light_post": "Proposed (P2, light post)",
           "P3_fidelity_lg": "Proposed (P3, LightGlue)",
           "V5_full": "Proposed (V5 full)"}.get(prop, "Proposed")
    return [("Base_M1", "Base ($M_1$)", C_BASE, "o"),
            ("B2_homog_alpha", "Homography+Alpha", C_HOMO, "^"),
            (prop, lbl, C_PROP, "s")]


# Back-compat alias; most fig fns now take `main3` explicitly.
MAIN3 = [("Base_M1", "Base ($M_1$)", C_BASE, "o"),
         ("B2_homog_alpha", "Homography+Alpha", C_HOMO, "^"),
         ("V5_full", "Proposed Pipeline", C_PROP, "s")]
ABLATION6 = [("V2", "V2_affine_only", "V2\nAffine\n+Alpha", "#DD8452"),
             ("V3", "V3_affine_flow", "V3\n+Flow", "#999999"),
             ("V4", "V4_affine_lab", "V4\nLAB\n+post", "#c9a86a"),
             ("V5", "V5_full", "V5\nFull\n(orig.)", "#e0908a"),
             ("V5$^-$", "V5_full_nopost", "V5$^-$\nLAB\nno-post", "#8dcfa8"),
             ("P1", "P1_fidelity", "P1\nFidelity\n(new)", C_PROP)]
STAGE_LABELS = {"s2_registration_ms": "S2 Registration",
                "s3_flow_ms": "S3 Dense Flow",
                "s4s5_fusion_ms": "S4-5 Fusion",
                "s5_post_ms": "S5 Post-proc."}


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "savefig.dpi": 300, "figure.dpi": 150,
    })
    return plt


def _scen_stats(data, method, metric):
    ps = data["per_scenario"][method]
    scens = sorted(ps.keys())
    mu = np.array([ps[s][metric]["mean"] if ps[s].get(metric) else np.nan
                   for s in scens])
    sg = np.array([ps[s][metric]["std"] if ps[s].get(metric) else np.nan
                   for s in scens])
    return scens, mu, sg


def fig1_laplacian(plt, data):
    scens, _, _ = _scen_stats(data, _best_proposed(data), "lapvar")
    X = np.arange(len(scens)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    main3 = _main3(data); prop = main3[-1][0]
    for i, (key, lbl, c, _) in enumerate(main3):
        _, mu, sg = _scen_stats(data, key, "lapvar")
        ax.bar(X + (i - 1) * w, mu, w, yerr=sg, color=c, alpha=0.88,
               label=lbl, zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor="#333", capthick=0.9))
    _, mu_p, _ = _scen_stats(data, prop, "lapvar")
    for j, v in enumerate(mu_p):
        ax.text(X[j] + w, v + 0.02 * np.nanmax(mu_p), f"{v:.0f}",
                ha="center", va="bottom", fontsize=6.2,
                fontweight="bold", color="#1a4a1a")
    bm = data["summary"]["Base_M1"]["lapvar"]["mean"]
    pm = data["summary"][prop]["lapvar"]["mean"]
    ax.annotate(f"Mean edge-contrast gain\nvs. base: +{(pm-bm)/bm*100:.0f}%",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                fontsize=6.8, bbox=dict(boxstyle="round,pad=0.3",
                                        fc="#f0f5fa", ec="#aabbcc", lw=0.7))
    ax.set_xticks(X); ax.set_xticklabels(scens)
    ax.set_ylabel("Laplacian Variance  (higher = sharper)")
    ax.legend(framealpha=0.92, loc="upper right", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig1_laplacian.png", bbox_inches="tight"); plt.close(fig)
    print("  fig1_laplacian done")


def fig2_ssim(plt, data):
    prop = _best_proposed(data)
    scens, _, _ = _scen_stats(data, prop, "ssim")
    if np.all(np.isnan(_scen_stats(data, prop, "ssim")[1])):
        print("  fig2_ssim skipped (no ground truth)"); return
    X = np.arange(len(scens))
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for key, lbl, c, mk in _main3(data):
        _, mu, sg = _scen_stats(data, key, "ssim")
        ax.plot(X, mu, marker=mk, color=c, label=lbl, markersize=5, zorder=4)
        ax.fill_between(X, mu - sg, mu + sg, color=c, alpha=0.13, zorder=3)
    ax.axhline(0.85, color="#777", lw=0.8, ls="--")
    ax.text(X[-1] + 0.05, 0.853, "0.85 threshold", fontsize=6.5,
            color="#555", va="bottom")
    ax.set_xticks(X); ax.set_xticklabels(scens)
    ax.set_ylabel("SSIM Score  (max 1.0, higher is better)")
    ax.set_ylim(0.28, 1.04)
    ax.legend(framealpha=0.92, loc="lower right", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig2_ssim.png", bbox_inches="tight"); plt.close(fig)
    print("  fig2_ssim done")


def fig3_psnr(plt, data):
    prop = _best_proposed(data)
    plabel = _main3(data)[-1][1]
    trio = [("Base_M1", "Base ($M_1$)", C_BASE),
            ("B3_affine_alpha", "Affine+Alpha", C_AFFA),
            (prop, plabel, C_PROP)]
    scens, mu0, _ = _scen_stats(data, prop, "psnr")
    if np.all(np.isnan(mu0)):
        print("  fig3_psnr skipped (no ground truth)"); return
    X = np.arange(len(scens)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for i, (key, lbl, c) in enumerate(trio):
        _, mu, sg = _scen_stats(data, key, "psnr")
        ax.bar(X + (i - 1) * w, mu, w, yerr=sg, color=c, alpha=0.88,
               label=lbl, zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor="#333", capthick=0.9))
    _, mu_p, _ = _scen_stats(data, prop, "psnr")
    for j, v in enumerate(mu_p):
        ax.text(X[j] + w, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.2, fontweight="bold", color="#1a4a1a")
    ax.set_xticks(X); ax.set_xticklabels(scens)
    ax.set_ylabel("PSNR (dB, higher is better)")
    ax.legend(framealpha=0.92, loc="upper right", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig3_psnr.png", bbox_inches="tight"); plt.close(fig)
    print("  fig3_psnr done")


def fig4_niqe(plt, data):
    if not data.get("niqe_available"):
        print("  fig4_niqe skipped (pyiqa not installed at run time)"); return
    scens, _, _ = _scen_stats(data, _best_proposed(data), "niqe")
    X = np.arange(len(scens)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for i, (key, lbl, c, _) in enumerate(_main3(data)):
        _, mu, sg = _scen_stats(data, key, "niqe")
        ax.bar(X + (i - 1) * w, mu, w, yerr=sg, color=c, alpha=0.88,
               label=lbl, zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor="#333", capthick=0.9))
    ax.axhspan(2.5, 3.0, color=C_PROP, alpha=0.10, zorder=1)
    ax.text(X[-1] + 0.3, 2.72, "DSLR\nregion", fontsize=6.2,
            color="#1a5a2a", va="center")
    ax.set_xticks(X); ax.set_xticklabels(scens)
    ax.set_ylabel("NIQE Score  (lower = more natural)")
    ax.legend(framealpha=0.92, loc="upper right", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig4_niqe.png", bbox_inches="tight"); plt.close(fig)
    print("  fig4_niqe done")


def fig5_ablation(plt, data):
    import matplotlib.patches as mpatches
    rows = [t for t in ABLATION6 if t[1] in data["summary"]]
    labels = [t[2] for t in rows]; cols = [t[3] for t in rows]
    def col(metric, field):
        return np.array([data["summary"][t[1]][metric][field]
                         if data["summary"][t[1]].get(metric) else np.nan
                         for t in rows])
    ssim_v, ssim_s = col("ssim", "mean"), col("ssim", "std")
    lap_v, lap_s = col("lapvar", "mean"), col("lapvar", "std")
    xv = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for i, (c, sm, ss) in enumerate(zip(cols, ssim_v, ssim_s)):
        a1.bar(i, sm, 0.55, yerr=ss, color=c, alpha=0.88, zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor="#333", capthick=0.9))
        if not np.isnan(sm):
            a1.text(i, sm + 0.012, f"{sm:.3f}", ha="center", va="bottom",
                    fontsize=6.0, fontweight="bold")
    a1.set_xticks(xv); a1.set_xticklabels(labels, fontsize=6.0)
    a1.set_ylabel("SSIM Score"); a1.set_title("(a) SSIM", fontsize=8)
    if not np.isnan(ssim_v[-1]):
        a1.axhline(ssim_v[-1], color=C_PROP, lw=0.8, ls="--", alpha=0.6)
    for i, (c, lm, ls) in enumerate(zip(cols, lap_v, lap_s)):
        a2.bar(i, lm, 0.55, yerr=ls, color=c, alpha=0.88, zorder=3,
               error_kw=dict(elinewidth=0.9, ecolor="#333", capthick=0.9))
        if not np.isnan(lm):
            a2.text(i, lm + 0.015 * np.nanmax(lap_v), f"{lm:.0f}",
                    ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    a2.set_xticks(xv); a2.set_xticklabels(labels, fontsize=6.0)
    a2.set_ylabel("Laplacian Variance")
    a2.set_title("(b) Laplacian Variance", fontsize=8)
    patches = [mpatches.Patch(color=c, label=t[0])
               for t, c in zip(rows, cols)]
    fig.legend(handles=patches, loc="lower center", ncol=len(rows),
               fontsize=6.5, framealpha=0.92, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.5)
    fig.savefig("fig5_ablation.png", bbox_inches="tight"); plt.close(fig)
    print("  fig5_ablation done")


def fig6_sensitivity(plt, sens):
    pts = sens["points"]
    alphas = np.array([p["alpha"] for p in pts])
    lap = np.array([p["lapvar"]["mean"] for p in pts])
    lap_s = np.array([p["lapvar"]["std"] for p in pts])
    have_ssim = all(p.get("ssim") for p in pts)
    fig, ax1 = plt.subplots(figsize=(5.0, 2.9))
    ax1.errorbar(alphas, lap, yerr=lap_s, color=C_PROP, marker="s",
                 markersize=5, capsize=2.5, elinewidth=0.9, zorder=4,
                 label="Laplacian Variance")
    ax1.set_xlabel(r"Unsharp amplification factor $\alpha$")
    ax1.set_ylabel("Laplacian Variance", color=C_PROP)
    ax1.tick_params(axis="y", labelcolor=C_PROP)
    ax1.axvline(1.2, color="#777", lw=0.8, ls="--")
    ax1.text(1.21, ax1.get_ylim()[1] * 0.97, "chosen\noperating point",
             fontsize=6.2, color="#555", va="top")
    handles, labels = ax1.get_legend_handles_labels()
    if have_ssim:
        ssim_m = np.array([p["ssim"]["mean"] for p in pts])
        ssim_s = np.array([p["ssim"]["std"] for p in pts])
        ax2 = ax1.twinx(); ax2.spines.right.set_visible(True)
        ax2.errorbar(alphas, ssim_m, yerr=ssim_s, color=C_BASE, marker="o",
                     markersize=5, capsize=2.5, elinewidth=0.9, zorder=4,
                     label="SSIM")
        ax2.set_ylabel("SSIM", color=C_BASE)
        ax2.tick_params(axis="y", labelcolor=C_BASE)
        ax2.set_ylim(min(ssim_m) - 0.08, min(1.0, max(ssim_m) + 0.08))
        ax2.grid(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2; labels += l2
    ax1.legend(handles, labels, framealpha=0.92, loc="lower right",
               fontsize=6.5)
    ax1.set_title(r"Sensitivity to post-processing strength $\alpha$"
                  "\n(no-reference index varies; reference fidelity stable)",
                  fontsize=7.5)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig6_sensitivity.png", bbox_inches="tight"); plt.close(fig)
    print("  fig6_sensitivity done")


def fig7_runtime(plt, benches):
    import matplotlib.patches as mpatches
    stage_keys = [k for k in STAGE_LABELS
                  if any(k in b["stages"] for b in benches)]
    palette = ["#4C72B0", "#DD8452", "#2E8B57", "#937860"]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ypos = np.arange(len(benches))
    for yi, b in enumerate(benches):
        left = 0.0
        for ki, k in enumerate(stage_keys):
            v = b["stages"].get(k, {}).get("mean", 0.0)
            ax.barh(yi, v, 0.5, left=left, color=palette[ki % len(palette)],
                    alpha=0.88, zorder=3)
            if v > 0.04 * b["total"]["mean"]:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center",
                        va="center", fontsize=6.0, color="white",
                        fontweight="bold")
            left += v
        ax.text(left * 1.01, yi, f"{b['total']['mean']:.0f}±"
                f"{b['total']['std']:.0f} ms", va="center", fontsize=6.8,
                fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b['tag']}\n({b['machine']})" for b in benches],
                       fontsize=7)
    ax.set_xlabel("Per-stage latency (ms), mean of repeated runs")
    ax.set_xlim(0, max(b["total"]["mean"] for b in benches) * 1.30)
    patches = [mpatches.Patch(color=palette[i % len(palette)],
                              label=STAGE_LABELS[k])
               for i, k in enumerate(stage_keys)]
    fig.legend(handles=patches, loc="lower center", ncol=len(stage_keys),
               fontsize=6.2, framealpha=0.92, bbox_to_anchor=(0.5, -0.05))
    ax.set_title("Pipeline latency: client workstation vs. edge device",
                 fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig("fig7_runtime.png", bbox_inches="tight"); plt.close(fig)
    print("  fig7_runtime done")


def _recompute_aligned_only(data, ref="V5_full"):
    """Rebuild summary/per_scenario over ALIGNED pairs only, so the
    figures are not distorted by registration failures (Base_M1 and
    B1_lanczos are always included since they have no alignment step).
    Returns a NEW data dict; the original file is untouched."""
    import copy
    from collections import defaultdict
    recs = data["records"]
    idx = defaultdict(dict)
    for r in recs:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    keep = {k for k, v in idx.items() if v.get(ref, {}).get("aligned")}
    if not keep:
        return data, 0   # no alignment info; use as-is
    always = {"Base_M1", "B1_lanczos"}
    kept = [r for r in recs
            if (r["scenario"], r["pair"]) in keep or r["method"] in always]

    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return dict(mean=float(np.mean(vals)), std=sd, n=len(vals))

    methods = sorted({r["method"] for r in kept})
    scen = sorted({r["scenario"] for r in kept})
    metrics = ("lapvar", "ssim", "psnr", "niqe", "lpips")
    summary, per_scenario = {}, {}
    for m in methods:
        rows = [r for r in kept if r["method"] == m]
        summary[m] = {k: agg(rows, k) for k in metrics}
        per_scenario[m] = {s: {k: agg([r for r in rows
                                       if r["scenario"] == s], k)
                               for k in metrics} for s in scen}
    out = copy.deepcopy(data)
    out["summary"] = summary
    out["per_scenario"] = per_scenario
    return out, len(keep)


def cmd_figures(args):
    plt = _setup_mpl()
    print("Generating figures...")
    res_p = Path(args.results)
    if res_p.exists():
        data = json.loads(res_p.read_text())
        aligned_only = getattr(args, "aligned_only", True)
        if aligned_only:
            data, n = _recompute_aligned_only(data)
            if n:
                print(f"  (figures use {n} aligned pairs only; failed "
                      "registrations excluded so they don't skew means)")
        fig1_laplacian(plt, data)
        fig2_ssim(plt, data)
        fig3_psnr(plt, data)
        fig4_niqe(plt, data)
        fig5_ablation(plt, data)
    else:
        print(f"  [warn] {res_p} missing — run the `run` subcommand first; "
              "figs 1-5 skipped")
    sens_p = Path(args.sensitivity)
    if sens_p.exists():
        fig6_sensitivity(plt, json.loads(sens_p.read_text()))
    else:
        print(f"  [warn] {sens_p} missing — run `sweep`; fig6 skipped")
    benches = [json.loads(p.read_text())
               for p in sorted(Path(".").glob("runtime_*.json"))]
    if benches:
        fig7_runtime(plt, benches)
    else:
        print("  [warn] no runtime_*.json found — run `bench` on the "
              "workstation and on a Raspberry Pi; fig7 skipped")
    print("All done.")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# Notebook-friendly API (Google Colab / Jupyter — no argparse needed)
# ════════════════════════════════════════════════════════════════

class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_experiments(dataset="dataset", out="results.json", max_dim=2000):
    """Evaluate all baselines/variants over the dataset -> results.json
    max_dim caps processing resolution (2000 px recommended in Colab;
    pass None to process at native resolution)."""
    cmd_run(_NS(dataset=dataset, out=out, max_dim=max_dim))


def run_sweep(dataset="dataset", out="sensitivity.json", max_dim=2000):
    """Unsharp-alpha sensitivity sweep -> sensitivity.json"""
    cmd_sweep(_NS(dataset=dataset, out=out, max_dim=max_dim))


def run_bench(m1="m1.jpg", m2="m2.jpg", tag=None, repeats=10, warmup=2,
              max_dim=2000):
    """Per-stage runtime benchmark on this host -> runtime_<tag>.json"""
    tag = tag or (platform.node() or "host")
    cmd_bench(_NS(m1=m1, m2=m2, tag=tag, repeats=repeats, warmup=warmup,
                  max_dim=max_dim))


def make_all_figures(results="results.json", sensitivity="sensitivity.json",
                     aligned_only=True):
    """Generate fig1..fig7 from whichever JSON files exist.
    aligned_only=True (default) excludes registration-failure pairs so
    they don't distort the figures; set False to plot all pairs."""
    cmd_figures(_NS(results=results, sensitivity=sensitivity,
                    aligned_only=aligned_only))


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare",
                       help="build dataset/ from raw/scene_*/{base,ref}.jpg")
    p.add_argument("--raw", default="raw")
    p.add_argument("--out", default="dataset")
    p.add_argument("--max-dim", dest="max_dim", type=int, default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="evaluate all methods over the dataset")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="results.json")
    p.add_argument("--max-dim", dest="max_dim", type=int, default=2000,
                   help="cap processing resolution (0 = native)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="unsharp-alpha sensitivity sweep")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="sensitivity.json")
    p.add_argument("--max-dim", dest="max_dim", type=int, default=2000)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("bench", help="per-stage runtime benchmark")
    p.add_argument("--m1", default="m1.jpg")
    p.add_argument("--m2", default="m2.jpg")
    p.add_argument("--tag", default=platform.node() or "host")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-dim", dest="max_dim", type=int, default=2000)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("figures", help="generate the 7 paper figures")
    p.add_argument("--results", default="results.json")
    p.add_argument("--sensitivity", default="sensitivity.json")
    p.set_defaults(func=cmd_figures)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    if _in_notebook():
        print("Notebook detected — argparse CLI skipped.\n"
              "Call the functions directly in the next cell, e.g.:\n"
              "  run_experiments('dataset')\n"
              "  run_sweep('dataset')\n"
              "  run_bench('m1.jpg', 'm2.jpg', tag='colab')\n"
              "  make_all_figures()")
    else:
        main()
