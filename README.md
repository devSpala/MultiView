# MultiView — When a Second View Helps

A study of **multi-view image enhancement** for degraded image recognition:
given a degraded base frame `M1` from one camera and a sharp reference `M2`
from a second camera with an overlapping field of view, how much does the
second view help, at what cost, and how does a lightweight classical
pipeline compare to a learned reference-based super-resolution (RefSR)
network on a downstream OCR task?

This repository contains the full pipeline, the interactive demo
(`app.py`), the evaluation harnesses, and the scripts that generate every
figure in the paper.

> **Headline finding.** A naive classical pipeline *degrades*
> quality (registration collapses on 38% of pairs; aggressive sharpening
> destroys fidelity). Two minimal fixes — learned matching (registration
> 34/55 → 55/55) and removing post-processing — restore it. Under a fair
> *matched-resolution* comparison, a **fidelity-versus-readability
> trade-off** emerges: classical fusion preserves fidelity (SSIM 0.81 vs.
> 0.70) and scales to full-resolution frames on commodity hardware, while
> learned RefSR (TTSR) wins downstream OCR readability at a fidelity cost,
> a GPU requirement, and an inability to process full-resolution frames.

---

## Repository structure

```
MultiView/
├── enhance_core.py                  # shared enhancement core (S1–S5 pipeline)
├── app.py                           # Streamlit demo (Enhance + Compare modes)
├── compare_headless.py              # matched-resolution evaluation harness
├── experimental_results_standalone.py   # metric + OCR pipeline helpers
├── ocr_downstream_v2.py             # OCR scoring helpers
├── refsr_ttsr_adapter.py            # TTSR RefSR baseline adapter
├── publication_figures.py           # generates the 10 main paper figures
├── refsr_figures.py                 # generates the 3 RefSR-comparison figures
├── results.json                     # full-resolution results (input to figures)
├── compare_proc512.json             # matched-resolution results (512 px)
├── runtime_*.json                   # per-stage latency benchmarks
├── dataset/                         # capture data (see "Dataset format")
└── paper/                           # LaTeX source, main.bib, Pics/
```

---

## Installation

Python 3.9+ is recommended. Install the core dependencies:

```bash
pip install numpy opencv-contrib-python pillow scipy matplotlib streamlit
```

`opencv-contrib-python` (not plain `opencv-python`) is required — the
edge-aware fusion uses `cv2.ximgproc.guidedFilter`. On a headless server use
`opencv-contrib-python-headless` instead.

Optional components, needed only for the learned baselines and metrics:

```bash
# learned matcher (LightGlue); the code falls back to SIFT if absent
pip install lightglue torch

# perceptual / no-reference metrics (LPIPS, NIQE)
pip install pyiqa

# downstream OCR
pip install pytesseract
sudo apt-get install -y tesseract-ocr      # Tesseract binary
```

The TTSR RefSR baseline is **not** bundled (it is a heavy GPU model); see
[Running the RefSR comparison](#3-matched-resolution-refsr-comparison-with-ttsr).

---

## 1. Interactive demo (`app.py`)

The Streamlit app runs the **same** `enhance_core` used for the paper
numbers, so what you see matches what the harness reports.

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`).

**Two modes:**

- **Enhance** — upload a degraded base (`M1`) and a sharp reference (`M2`),
  toggle the algorithm options (LightGlue vs. SIFT matcher, multi-scale
  pyramid fusion, guided-filter weighting, optional post-processing), and
  see the enhanced result plus an automatic A/B against the original
  single-scale fusion. If you also upload a ground-truth frame, SSIM/PSNR
  are reported.
- **Compare vs RefSR** — runs the proposed method and TTSR side by side at a
  matched resolution and shows the fidelity-vs-readability table. Both
  methods are cropped to the same ROI and scored identically, so the
  comparison cannot be gamed.

To enable the live TTSR baseline in the app, set it in your launch
environment before starting Streamlit:

```python
import streamlit as st
from refsr_ttsr_adapter import make_ttsr_adapter
st.session_state["ttsr_fn"] = make_ttsr_adapter(
    "TTSR", "TTSR/TTSR.pt", "cuda", 4, "pm1", max_proc=384)
```

For the actual paper numbers (with LPIPS/NIQE/OCR), use the headless harness
below rather than the app.

---

## Dataset format

The evaluation harness iterates over `dataset/<scenario>/<pair>/`. Each pair
directory holds the degraded base, the sharp reference, the pixel-aligned
ground-truth frame, and a human transcription of the visible text:

```
dataset/
├── S1_gaussian_blur/
│   ├── scene01/
│   │   ├── M1.png        # degraded base frame
│   │   ├── M2.png        # sharp reference (second camera)
│   │   ├── gt.png        # pixel-aligned ground truth (sharp base)
│   │   └── gt.txt        # human transcription of visible text
│   └── scene02/ ...
├── S2_strong_blur/ ...
├── S3_motion_blur/ ...
├── S4_motion_jpeg/ ...
└── S5_lowlight_noise/ ...
```

The five scenarios span the dominant failure causes of constrained capture
(mild/strong Gaussian blur, motion blur, motion+JPEG, low-light+noise). With
11 scenes × 5 scenarios this yields **55 degraded–reference pairs**, each
with a true ground-truth frame.

---

## Reproducing the experiments

There are two evaluation passes. The **full-resolution fidelity** pass
produces `results.json` (used by the main figures and Table 1). The
**matched-resolution** pass produces `compare_proc512.json` (used by the
RefSR figures and Table 2, including the TTSR baseline).

### 2. Full-resolution fidelity evaluation

This pass runs the classical variants over the aligned pairs and records
SSIM / PSNR / LPIPS / NIQE / edge-contrast. Import `pyiqa` **before** the
pipeline so LPIPS and NIQE are available:

```python
import pyiqa                                 # MUST be imported first
import experimental_results_standalone as pipe
print("LPIPS:", pipe.HAVE_LPIPS, "| NIQE:", pipe.HAVE_NIQE)   # both True

pipe.run("dataset")                          # writes results.json
```

### 3. Matched-resolution RefSR comparison (with TTSR)

This is the fair comparison: every method (Input, Proposed single-scale,
Proposed multi-scale, TTSR) is cropped to the same ROI and resized to the
same resolution before scoring. It runs best on a GPU runtime (e.g. Google
Colab) because of the learned matcher and TTSR.

**Set up TTSR** (one-time): clone the TTSR repository into `./TTSR` and place
the pretrained reconstruction weights at `TTSR/TTSR.pt`.

```bash
git clone https://github.com/researchmm/TTSR.git
# download the reconstruction-trained weights into TTSR/TTSR.pt
```

**Run the harness** (Colab cell or script). Import order matters — `pyiqa`
before the pipeline:

```python
import pyiqa                                 # MUST be imported first
import experimental_results_standalone, ocr_downstream_v2
from refsr_ttsr_adapter import make_ttsr_adapter
import compare_headless as ch

ttsr = make_ttsr_adapter("TTSR", "TTSR/TTSR.pt", "cuda", 4, "pm1",
                         max_proc=384)        # cap internal res to avoid OOM
summary = ch.run("dataset", proc=512, ttsr_adapter=ttsr, gt_txt=True)
```

This writes `compare_proc512.json` and prints the matched-resolution table
plus paired Wilcoxon significance tests. If you hit CUDA OOM, lower
`max_proc` to 256 (this only shrinks TTSR's internal processing size, not the
evaluation resolution).

To run the classical-only comparison without TTSR, simply omit the adapter:

```bash
python compare_headless.py        # uses dataset/, proc=512, no TTSR
```

---

## 4. Generating the figures

All figures are produced from the result JSONs — no GPU, dataset, or TTSR
needed at this stage.

```bash
# 10 main figures (metrics, alignment, heatmap, ssim box, lapvar box,
# radar, tradeoff, win-rate, quality gain, runtime) from results.json
python publication_figures.py --errorbars off

# 3 RefSR-comparison figures (fig_ocr_3way, fig_refsr_tradeoff,
# fig_refsr_metrics) from compare_proc512.json
python refsr_figures.py --results compare_proc512.json --errorbars off
```

The `--errorbars off` flag disables error bars (used for the camera-ready
figures). `fig_runtime` additionally requires one or more `runtime_*.json`
benchmark files in the working directory.

---

## Results

**Table 1 — Aggregate fidelity over aligned pairs (full resolution).** Each
row is averaged over the pairs that method registered; the proposed method
registers all 55.

| Method                | SSIM ↑ | PSNR ↑ | LPIPS ↓ | NIQE ↓ | Align. |
|-----------------------|:------:|:------:|:-------:|:------:|:------:|
| Degraded input        | 0.819  | 22.77  | 0.376   | 9.48   | –      |
| Homography + blend    | 0.748  | 17.39  | 0.414   | 5.68   | 34/55  |
| Aggressive sharpening | 0.656  | 18.65  | 0.575   | 5.09   | 34/55  |
| **Proposed**          | 0.809  | 22.72  | 0.367   | 5.91   | 55/55  |

**Table 2 — Matched-resolution comparison at 512 px (n = 55).** Every method
is cropped to the same ROI and resized identically before scoring.

| Method                  | SSIM ↑ | PSNR ↑ | LPIPS ↓ | NIQE ↓ | OCR-c ↑ | OCR-w ↑ |
|-------------------------|:------:|:------:|:-------:|:------:|:-------:|:-------:|
| Input                   | 0.816  | 23.04  | 0.310   | 6.71   | 0.169   | 0.044   |
| Proposed (single)       | 0.806  | 22.98  | 0.299   | 5.31   | 0.170   | 0.037   |
| Proposed (multi-scale)  | 0.805  | 22.93  | 0.312   | 5.37   | 0.163   | 0.034   |
| TTSR-rec                | 0.696  | 18.76  | 0.279   | 5.03   | 0.192   | 0.002   |

Paired OCR (character accuracy) vs. input: Proposed (single) 31/55,
p = 0.39 (n.s.); TTSR 40/55, p = 0.067 (n.s. at α = 0.05). At matched
resolution neither classical fusion nor TTSR significantly changes OCR over
the input — the trade-off is in *fidelity vs. perceptual readability*, not a
significant downstream win.

---

## Pipeline overview

The enhancement core (`enhance_core.py`) is five stages:

| Stage | Description |
|-------|-------------|
| **S1 Registration** | match `M1`↔`M2` (LightGlue or SIFT) and estimate a partial-affine transform under RANSAC, computed on downscaled copies and rescaled to full resolution |
| **S2 Dense flow** | refine residual misalignment with a dense optical-flow field (classical Farnebäck or RAFT), using a structure-emphasising high-pass pre-filter |
| **S3 Sharpness weighting** | Laplacian-variance map favouring the reference where it is sharper, optionally refined edge-aware with a guided filter |
| **S4 Luminance fusion** | transfer reference detail in the CIELAB `L` channel only (single-scale, or multi-scale Laplacian pyramid), preserving `M1` chrominance |
| **S5 Post-processing** | *optional* CLAHE + unsharp masking; **omitted** in the proposed configuration (it inflates edge contrast while destroying fidelity) |

The recommended configuration uses the learned matcher and disables S5.

---

## Troubleshooting

- **LPIPS / NIQE come back empty** — `pyiqa` must be imported **before**
  `experimental_results_standalone` (the pipeline reads metric availability
  at import time). Restart the runtime and import `pyiqa` first.
- **`cv2.ximgproc` not found** — install `opencv-contrib-python`
  (or `-headless`), not plain `opencv-python`.
- **TTSR CUDA out of memory** — its dense patch-matching scales with the
  fourth power of the input side; lower `max_proc` (e.g. 256) when building
  the adapter. TTSR cannot process full-resolution frames on a 14 GB GPU.
- **LightGlue not installed** — the matcher falls back to SIFT automatically;
  results use whichever matcher is available.
- **Tesseract errors** — install the `tesseract-ocr` system binary in
  addition to the `pytesseract` Python package.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{multiview2027,
  title     = {When a Second View Helps: A Study of Multi-View Enhancement
               for Degraded Image Recognition},
  author    = {Anonymous},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

---

## License

Released under the MIT License unless noted otherwise. Third-party models
(TTSR, LightGlue, RAFT) remain under their respective licenses.
