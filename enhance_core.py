"""
enhance_core.py — improved multi-view enhancement (shared core)
================================================================
Used by BOTH the Streamlit app and the headless paper-evaluation
harness, so the interactive demo and the reported numbers come from
IDENTICAL code (a reproducibility requirement).

Genuine algorithmic improvements over the v5 pipeline, each defensible
on its own merits (NOT evaluation tuning):

  (I1) Pluggable matcher: LightGlue when available, SIFT fallback.
       LightGlue is what the paper's "proposed" method uses; the demo
       was still on SIFT. Same 4-DOF partial-affine estimation.

  (I2) Multi-scale Laplacian-pyramid detail transfer, replacing the
       single Gaussian-scale extraction in lab_frequency_fusion. Detail
       is transferred at several frequency bands, which improves both
       fidelity (finer bands) and legibility (mid bands) without
       over-amplifying any single scale.

  (I3) Guided-filter refinement of the sharpness weight map (edge-aware),
       replacing the Gaussian blur that bleeds weights across text
       boundaries. Respects character edges -> cleaner fusion.

All three compute the same way regardless of which method "wins" any
metric; none selects evaluation conditions.
"""

import numpy as np
import cv2

# optional edge-aware guided filter
try:
    import cv2.ximgproc as xip
    _HAVE_GUIDED = True
except Exception:
    _HAVE_GUIDED = False

# optional learned matcher (LightGlue + SuperPoint via the lightglue pkg)
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    _LG_EXTRACTOR = None
    _LG_MATCHER = None
    _HAVE_LIGHTGLUE = True
except Exception:
    _HAVE_LIGHTGLUE = False


REG_MAX_DIM = 1600


# ── matcher: LightGlue (preferred) or SIFT (fallback) ────────────────
def _scaled_gray(img, max_dim):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return g, 1.0
    s = max_dim / m
    g = cv2.resize(g, (int(round(w*s)), int(round(h*s))),
                   interpolation=cv2.INTER_AREA)
    return g, s


def _lightglue_points(g1, g2, device="cpu"):
    """Return matched (src_pts in g2, dst_pts in g1) using LightGlue."""
    global _LG_EXTRACTOR, _LG_MATCHER
    if _LG_EXTRACTOR is None:
        _LG_EXTRACTOR = SuperPoint(max_num_keypoints=2048).eval().to(device)
        _LG_MATCHER = LightGlue(features="superpoint").eval().to(device)

    def _t(g):
        t = torch.from_numpy(g).float()[None, None] / 255.0
        return t.to(device)

    with torch.no_grad():
        f1 = _LG_EXTRACTOR.extract(_t(g1))
        f2 = _LG_EXTRACTOR.extract(_t(g2))
        m = _LG_MATCHER({"image0": f2, "image1": f1})  # match g2 -> g1
        f2, f1, m = [rbd(x) for x in (f2, f1, m)]
    idx = m["matches"]
    src = f2["keypoints"][idx[:, 0]].cpu().numpy()
    dst = f1["keypoints"][idx[:, 1]].cpu().numpy()
    return src, dst


def stable_alignment(m1, m2, max_features=4000, matcher="auto", device="cpu"):
    """4-DOF partial-affine aligning m2 -> m1 in FULL-res coords.
    matcher: 'auto' (LightGlue if available else SIFT), 'lightglue', 'sift'.
    Returns (M 2x3, n_inliers, matcher_used). M is None on failure."""
    g1, s1 = _scaled_gray(m1, REG_MAX_DIM)
    g2, s2 = _scaled_gray(m2, REG_MAX_DIM)

    use_lg = matcher == "lightglue" or (matcher == "auto" and _HAVE_LIGHTGLUE)
    used = "none"

    def _finish(src_pts, dst_pts):
        if src_pts is None or len(src_pts) < 10:
            return None, len(src_pts) if src_pts is not None else 0
        src = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
        M, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=5000, confidence=0.995, refineIters=10)
        if M is None:
            return None, 0
        scale = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))
        if not (0.1 <= scale <= 10.0):
            return None, 0
        n = int(mask.sum()) if mask is not None else 0
        # rescale to full-res
        S1i = np.diag([1.0/s1, 1.0/s1, 1.0])
        S2 = np.diag([s2, s2, 1.0])
        M3 = np.vstack([M, [0, 0, 1.0]])
        Mf = (S1i @ M3 @ S2)[:2, :]
        return Mf.astype(np.float64), n

    if use_lg:
        try:
            src, dst = _lightglue_points(g1, g2, device)
            M, n = _finish(src, dst)
            if M is not None:
                return M, n, "lightglue"
        except Exception:
            pass  # fall through to SIFT

    # SIFT fallback (deterministic)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0, "sift"
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(d2, d1, k=2)
    for ratio in (0.70, 0.65):
        good = [a for a, b in pairs if a.distance < ratio * b.distance]
        if len(good) < 10:
            continue
        src = np.float32([kp2[a.queryIdx].pt for a in good])
        dst = np.float32([kp1[a.trainIdx].pt for a in good])
        M, n = _finish(src, dst)
        if M is not None:
            return M, n, "sift"
    return None, 0, "sift"


def warp_aligned(img, M, target_shape):
    h, w = target_shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


def auto_roi_from_affine(M, m2_shape, m1_shape, margin=8):
    h2, w2 = m2_shape[:2]; h1, w1 = m1_shape[:2]
    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    M3 = np.vstack([M, [0,0,1]]).astype(np.float32)
    proj = cv2.perspectiveTransform(corners, M3).reshape(-1,2)
    x1 = max(int(np.ceil(proj[:,0].min()))+margin, 0)
    y1 = max(int(np.ceil(proj[:,1].min()))+margin, 0)
    x2 = min(int(np.floor(proj[:,0].max()))-margin, w1)
    y2 = min(int(np.floor(proj[:,1].max()))-margin, h1)
    if x2-x1 < 32 or y2-y1 < 32:
        return None
    return (x1, y1, x2, y2)


# ── flow refinement (unchanged structure) ────────────────────────────
def _structure_image(gray):
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    low = cv2.GaussianBlur(eq, (0,0), 5.0)
    hp = cv2.subtract(eq, low)
    return cv2.add(hp, 128)


def refine_with_flow(img1, warped_img2, prefilter=True):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    if prefilter:
        g1, g2 = _structure_image(g1), _structure_image(g2)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g1.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(warped_img2, x-flow[...,0], y-flow[...,1],
                     cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


# ── (I3) sharpness weighting with guided-filter refinement ───────────
def get_sharpness_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mu = cv2.GaussianBlur(lap, (21,21), 0)
    mu2 = cv2.GaussianBlur(lap*lap, (21,21), 0)
    return np.maximum(0.0, mu2 - mu*mu)


def sharpness_weight(roi_m1, roi_m2, guided=True):
    """W in [0,1] favouring the reference where it is sharper. With
    guided=True the map is edge-aware-refined against the base image so
    weights respect text boundaries instead of bleeding across them."""
    s1 = get_sharpness_map(roi_m1)
    s2 = get_sharpness_map(roi_m2)
    diff = np.clip(s2 - s1, 0, None)
    mx = float(diff.max())
    w = diff/mx if mx > 1e-5 else np.zeros_like(diff, dtype=np.float32)
    w = w.astype(np.float32)
    if guided and _HAVE_GUIDED:
        guide = cv2.cvtColor(roi_m1, cv2.COLOR_BGR2GRAY)
        try:
            wg = xip.guidedFilter(guide, w, radius=16, eps=1e-2)
            # guided filter divides by local variance; perfectly uniform
            # guide regions can yield NaN/Inf -> sanitize and fall back.
            if np.isfinite(wg).all():
                w = np.clip(wg, 0, 1)
            else:
                w = cv2.GaussianBlur(w, (31, 31), 0)
        except Exception:
            w = cv2.GaussianBlur(w, (31, 31), 0)
    else:
        w = cv2.GaussianBlur(w, (31, 31), 0)
    return np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


# ── (I2) multi-scale Laplacian-pyramid detail transfer ───────────────
def _laplacian_pyramid(img32, levels):
    """Laplacian pyramid robust to odd dimensions (pyrUp restores the
    exact size of the finer level via dstsize)."""
    g = img32.copy()
    gp = [g]
    for _ in range(levels):
        g = cv2.pyrDown(g)
        gp.append(g)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def lab_pyramid_fusion(roi_blurry, roi_sharp, weight_map, levels=3,
                       multiscale=True):
    """Transfer reference L*-detail into the base. multiscale=True uses a
    Laplacian pyramid (several frequency bands); False reproduces the
    original single-scale behaviour for ablation."""
    lab_b = cv2.cvtColor(roi_blurry, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(roi_sharp,  cv2.COLOR_BGR2LAB).astype(np.float32)
    l_b, a_b, b_b = cv2.split(lab_b)
    l_s, _, _ = cv2.split(lab_s)
    w = weight_map.astype(np.float32)

    if not multiscale:
        l_s_smooth = cv2.GaussianBlur(l_s, (15,15), 0)
        detail = l_s - l_s_smooth
        enhanced_l = np.clip(l_b + detail * w, 0, 255)
    else:
        # transfer the reference's high/mid-frequency bands, keep the
        # base's lowest band (its global tone/structure -> fidelity).
        lp_s = _laplacian_pyramid(l_s, levels)
        lp_b = _laplacian_pyramid(l_b, levels)
        out_bands = []
        for i in range(levels):
            wl_i = cv2.resize(w, (lp_b[i].shape[1], lp_b[i].shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            out_bands.append(lp_b[i] * (1.0 - wl_i) + lp_s[i] * wl_i)
        out_bands.append(lp_b[levels])  # base lowest band -> preserves tone
        # collapse pyramid (use each level's stored shape as dstsize)
        cur = out_bands[levels]
        for i in range(levels-1, -1, -1):
            cur = cv2.pyrUp(cur, dstsize=(out_bands[i].shape[1],
                                          out_bands[i].shape[0]))
            cur = cur + out_bands[i]
        enhanced_l = np.clip(np.nan_to_num(cur), 0, 255)

    res = cv2.merge([enhanced_l, a_b, b_b]).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_LAB2BGR)


def postprocess(roi, clahe_clip, sharp_str):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8)).apply(l)
    roi = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(roi, (0,0), 1.5)
    return cv2.addWeighted(roi, 1.0+sharp_str, blur, -sharp_str, 0)


# ── full pipeline (one call, used by app + harness) ──────────────────
def enhance(m1, m2, matcher="auto", use_flow=True, prefilter=True,
            multiscale=True, guided=True, use_post=False,
            clahe_clip=2.5, sharp_str=1.2, device="cpu"):
    """Returns dict: result(full frame), aligned, roi, matcher_used,
    n_inliers, weight_map. result falls back to m1 if registration fails."""
    M, n_in, used = stable_alignment(m1, m2, matcher=matcher, device=device)
    if M is None:
        return dict(result=m1.copy(), aligned=False, roi=None,
                    matcher_used=used, n_inliers=n_in, weight=None)
    warped = warp_aligned(m2, M, m1.shape)
    if use_flow:
        warped = refine_with_flow(m1, warped, prefilter)
    roi = auto_roi_from_affine(M, m2.shape, m1.shape)
    if roi is None:
        roi = (0, 0, m1.shape[1], m1.shape[0])
    x1, y1, x2, y2 = roi
    roi_m1 = m1[y1:y2, x1:x2]; roi_m2 = warped[y1:y2, x1:x2]
    w = sharpness_weight(roi_m1, roi_m2, guided=guided)
    enh = lab_pyramid_fusion(roi_m1, roi_m2, w, multiscale=multiscale)
    if use_post:
        enh = postprocess(enh, clahe_clip, sharp_str)
    # soft-blend back
    h1, w1 = m1.shape[:2]
    mask = np.zeros((h1, w1), np.float32); mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    full = m1.copy(); full[y1:y2, x1:x2] = enh
    m3 = cv2.merge([mask, mask, mask])
    result = (m1.astype(np.float32)*(1-m3) + full.astype(np.float32)*m3
              ).astype(np.uint8)
    return dict(result=result, aligned=True, roi=roi, matcher_used=used,
                n_inliers=n_in, weight=w)
