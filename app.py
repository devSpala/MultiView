"""
app.py — Multi-View Enhancement Studio (v6)
===========================================
Interactive demo built on the SAME enhance_core used for the paper
numbers, so what you see here matches what compare_headless.py reports.

Two modes:
  1. Enhance  — run the improved method on an uploaded M1/M2 pair, with
                live toggles for the genuine algorithmic improvements
                (LightGlue matcher, multi-scale pyramid fusion, guided
                weighting) and an A/B against the original single-scale
                fusion so the improvement is visible.
  2. Compare  — run the proposed method AND a learned RefSR baseline
                (TTSR) side by side, scoring fidelity vs. readability so
                the honest trade-off is explicit: classical fusion buys
                fidelity + GPU-free deployment; learned RefSR buys
                readability at higher cost.

The Compare mode does NOT pick whichever setting makes one method win;
both methods are scored identically at the same resolution.
"""

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

import enhance_core as ec

st.set_page_config(page_title="Multi-View Enhancement Studio", layout="wide")


# ── helpers ──────────────────────────────────────────────────────────
def load_image(file) -> np.ndarray:
    pil = ImageOps.exif_transpose(Image.open(file))
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_max(img, max_dim):
    if img is None or not max_dim:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    s = max_dim / m
    return cv2.resize(img, (int(round(w*s)), int(round(h*s))),
                      interpolation=cv2.INTER_AREA)


def lapvar(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mu1, mu2 = a.mean(), b.mean()
    v1, v2 = a.var(), b.var()
    cov = ((a-mu1)*(b-mu2)).mean()
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return float(((2*mu1*mu2+c1)*(2*cov+c2)) /
                 ((mu1**2+mu2**2+c1)*(v1+v2+c2)))


def psnr(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return float(99.0 if mse < 1e-9 else 10*np.log10(255.0**2/mse))


@st.cache_data(show_spinner=False, max_entries=4)
def _enhance_cached(m1_b, m2_b, max_dim, matcher, use_flow, prefilter,
                    multiscale, guided, use_post, clahe, sharp):
    m1 = resize_max(load_image(io.BytesIO(m1_b)), max_dim)
    m2 = resize_max(load_image(io.BytesIO(m2_b)), max_dim)
    r = ec.enhance(m1, m2, matcher=matcher, use_flow=use_flow,
                   prefilter=prefilter, multiscale=multiscale, guided=guided,
                   use_post=use_post, clahe_clip=clahe, sharp_str=sharp)
    return m1, m2, r


# ── sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Multi-View Enhancement Studio")
mode = st.sidebar.radio("Mode", ["Enhance", "Compare vs RefSR"])

st.sidebar.header("Processing")
max_dim = st.sidebar.select_slider("Max working resolution (px)",
                                   [768, 1024, 1600, 2000], 1600)
matcher = st.sidebar.selectbox(
    "Matcher", ["auto", "sift", "lightglue"],
    help="auto uses LightGlue if installed, else SIFT. The paper's "
         "proposed method uses LightGlue.")
st.sidebar.caption(f"LightGlue installed: {ec._HAVE_LIGHTGLUE} · "
                   f"Guided filter: {ec._HAVE_GUIDED}")

st.sidebar.header("Algorithm improvements")
multiscale = st.sidebar.checkbox(
    "Multi-scale pyramid fusion", True,
    help="Transfer reference detail across several frequency bands "
         "(vs. single Gaussian scale). Improves fidelity + legibility.")
guided = st.sidebar.checkbox(
    "Guided-filter weighting", True,
    help="Edge-aware refinement of the sharpness weight map so weights "
         "respect text boundaries instead of bleeding across them.")
use_flow = st.sidebar.checkbox("Optical-flow refinement", True)
prefilter = st.sidebar.checkbox("Structure-guided flow", True)

st.sidebar.header("Post-processing (optional)")
use_post = st.sidebar.checkbox("CLAHE + unsharp", False,
                               help="Off by default: the paper shows it "
                                    "hurts fidelity and OCR.")
clahe = st.sidebar.slider("CLAHE clip", 1.0, 5.0, 2.5, disabled=not use_post)
sharp = st.sidebar.slider("Unsharp α", 0.0, 3.0, 1.2, disabled=not use_post)


# ── uploaders ────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    f1 = st.file_uploader("M1 — degraded base", ["jpg", "jpeg", "png"])
with c2:
    f2 = st.file_uploader("M2 — sharp reference", ["jpg", "jpeg", "png"])

gt_file = None
if mode == "Compare vs RefSR" or st.sidebar.checkbox(
        "I have ground truth (for metrics)", False):
    gt_file = st.file_uploader("Ground truth (optional, enables SSIM/PSNR)",
                               ["jpg", "jpeg", "png"])

if not (f1 and f2):
    st.info("Upload a degraded base (M1) and a sharp reference (M2) to begin.")
    st.stop()

m1_b, m2_b = f1.getvalue(), f2.getvalue()


# ── ENHANCE MODE ─────────────────────────────────────────────────────
if mode == "Enhance":
    with st.spinner("Aligning and fusing…"):
        m1, m2, r = _enhance_cached(m1_b, m2_b, max_dim, matcher, use_flow,
                                    prefilter, multiscale, guided, use_post,
                                    clahe, sharp)
    if not r["aligned"]:
        st.error(f"Registration failed (matcher={r['matcher_used']}, "
                 f"inliers={r['n_inliers']}). Try LightGlue or a closer "
                 f"reference view.")
        st.stop()

    st.success(f"Aligned · matcher={r['matcher_used']} · "
               f"inliers={r['n_inliers']}")
    g = cv2.cvtColor(load_image(io.BytesIO(gt_file.getvalue())),
                     cv2.COLOR_BGR2BGR) if gt_file else None

    col = st.columns(3)
    col[0].image(to_rgb(m1), caption="M1 (degraded)", use_container_width=True)
    col[1].image(to_rgb(m2), caption="M2 (reference)", use_container_width=True)
    col[2].image(to_rgb(r["result"]), caption="Enhanced",
                 use_container_width=True)

    # A/B vs original single-scale fusion to show the improvement
    with st.spinner("Computing single-scale baseline for A/B…"):
        _, _, r_old = _enhance_cached(m1_b, m2_b, max_dim, matcher, use_flow,
                                      prefilter, False, False, use_post,
                                      clahe, sharp)
    st.subheader("Improvement over original single-scale fusion")
    mcols = st.columns(3)
    mcols[0].metric("Edge contrast (input)", f"{lapvar(m1):.0f}")
    mcols[1].metric("Single-scale", f"{lapvar(r_old['result']):.0f}")
    mcols[2].metric("Multi-scale + guided", f"{lapvar(r['result']):.0f}")
    if g is not None:
        g_roi = None
        x1, y1, x2, y2 = r["roi"]
        g_roi = g[y1:y2, x1:x2]
        s_cols = st.columns(3)
        s_cols[0].metric("SSIM input",
                         f"{ssim(m1[y1:y2,x1:x2], g_roi):.4f}")
        s_cols[1].metric("SSIM single-scale",
                         f"{ssim(r_old['result'][y1:y2,x1:x2], g_roi):.4f}")
        s_cols[2].metric("SSIM multi-scale",
                         f"{ssim(r['result'][y1:y2,x1:x2], g_roi):.4f}")

    if r["weight"] is not None:
        wv = cv2.applyColorMap((r["weight"]*255).astype(np.uint8),
                               cv2.COLORMAP_JET)
        st.image(to_rgb(wv), caption="Sharpness weight map "
                 "(red = trust reference)", use_container_width=True)


# ── COMPARE MODE ─────────────────────────────────────────────────────
else:
    st.subheader("Proposed (classical) vs learned RefSR — honest trade-off")
    st.caption("Both methods scored identically at the same resolution. "
               "The point is the trade-off, not a single winner.")

    proc = st.slider("Comparison resolution (px, long side)", 256, 1024, 512,
                     help="Both methods are evaluated at this size. TTSR "
                          "cannot process full-res frames, so a matched "
                          "reduced resolution is the only fair comparison.")

    # TTSR is optional and loaded lazily; if the user hasn't wired it up,
    # we explain rather than fabricate a baseline.
    ttsr_fn = st.session_state.get("ttsr_fn", None)
    st.caption("TTSR baseline: " + ("loaded" if ttsr_fn else
               "not loaded — see note below"))

    if st.button("Run comparison"):
        m1 = resize_max(load_image(io.BytesIO(m1_b)), max_dim)
        m2 = resize_max(load_image(io.BytesIO(m2_b)), max_dim)
        g = load_image(io.BytesIO(gt_file.getvalue())) if gt_file else None

        with st.spinner("Running proposed method…"):
            r = ec.enhance(m1, m2, matcher=matcher, multiscale=multiscale,
                           guided=guided, use_post=False)
        roi = r["roi"] or (0, 0, m1.shape[1], m1.shape[0])
        x1, y1, x2, y2 = roi

        def rs(img):
            s = proc / max(img.shape[:2])
            return cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)))

        prop_roi = rs(r["result"][y1:y2, x1:x2])
        in_roi = rs(m1[y1:y2, x1:x2])
        outs = {"Input": in_roi, "Proposed": prop_roi}

        if ttsr_fn:
            with st.spinner("Running TTSR…"):
                ttsr_full = ttsr_fn(m1, m2)
            if ttsr_full.shape[:2] != m1.shape[:2]:
                ttsr_full = cv2.resize(ttsr_full,
                                       (m1.shape[1], m1.shape[0]))
            outs["TTSR-rec"] = rs(ttsr_full[y1:y2, x1:x2])

        cols = st.columns(len(outs))
        for (name, img), c in zip(outs.items(), cols):
            c.image(to_rgb(img), caption=name, use_container_width=True)

        # metrics table
        st.subheader("Fidelity vs. readability")
        rows = []
        g_roi = rs(g[y1:y2, x1:x2]) if g is not None else None
        for name, img in outs.items():
            row = {"Method": name, "Edge contrast": f"{lapvar(img):.0f}"}
            if g_roi is not None:
                row["SSIM↑"] = f"{ssim(img, g_roi):.4f}"
                row["PSNR↑"] = f"{psnr(img, g_roi):.2f}"
            rows.append(row)
        st.table(rows)
        st.info("Interpretation: classical fusion typically wins **fidelity** "
                "(SSIM/PSNR) and runs GPU-free; learned RefSR typically wins "
                "**readability** (sharper text) at higher compute. Report "
                "both — that frontier is the contribution.")

    with st.expander("How to load the TTSR baseline"):
        st.markdown(
            "TTSR isn't bundled (heavy GPU model). To enable the live "
            "baseline, in a launching script set:\n"
            "```python\n"
            "import streamlit as st\n"
            "from refsr_ttsr_adapter import make_ttsr_adapter\n"
            "st.session_state['ttsr_fn'] = make_ttsr_adapter(\n"
            "    'TTSR','TTSR/TTSR.pt','cuda',4,'pm1',max_proc=384)\n"
            "```\n"
            "For the paper numbers use `compare_headless.py` instead — it "
            "scores all methods at one resolution with LPIPS/NIQE/OCR.")
