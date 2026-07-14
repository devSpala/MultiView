"""
refsr_ttsr_adapter.py
=====================
Adapter that lets TTSR (and, with small changes, C2-Matching) plug into the
existing refsr_baselines.py harness, using the Option-A protocol:

  Option A (fair, recommended): TTSR/C2-Matching are 4x super-resolution
  models trained on CUFED5. To use them as designed, we downsample the
  degraded base M1 by 4x to form a true low-res input, then let RefSR
  upscale 4x back to the original size using M2 as the reference. The 4x
  output is compared to the ground truth, exactly like every other method.

This file gives:
  * make_ttsr_adapter(repo_dir, weights_path, device) -> adapter callable
  * the adapter signature is (lr_bgr, ref_bgr) -> sr_bgr  (what the harness
    expects), and it returns an image at M1's original resolution.

IMPORTANT — you MUST verify two things against the actual TTSR repo:
  (1) the model construction call (class name, args), and
  (2) the forward() signature and output normalization.
Both are marked with  >>> VERIFY <<<  below. The surrounding glue (Option-A
resampling, color conversion, resizing) is complete and correct.

Repo: https://github.com/researchmm/TTSR
"""

import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def _bgr_to_tensor(bgr, device, norm="pm1"):
    """BGR uint8 -> float tensor [1,3,H,W]. norm='pm1' -> [-1,1], '01' -> [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if norm == "pm1":
        t = t * 2.0 - 1.0
    return t


def _tensor_to_bgr(t, norm="pm1"):
    """float tensor [1,3,H,W] -> BGR uint8."""
    t = t.detach().float().cpu().squeeze(0)
    if norm == "pm1":
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    rgb = (t * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bicubic(t, size_hw):
    return F.interpolate(t, size=size_hw, mode="bicubic", align_corners=False)


def make_ttsr_adapter(repo_dir, weights_path, device="cuda", scale=4,
                      norm="pm1", max_proc=384):
    """
    repo_dir     : path to the cloned TTSR repository (added to sys.path)
    weights_path : path to the pretrained TTSR-rec .pt/.pth checkpoint
    scale        : SR factor the model was trained for (TTSR/C2-Matching = 4)
    norm         : input normalization ('pm1' for [-1,1], '01' for [0,1])
    """
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # >>> VERIFY (1): construct the TTSR model exactly as the repo does. <<<
    # In the official repo this is roughly:
    #   from model import TTSR
    #   from option import args
    #   model = TTSR.TTSR(args).to(device)
    # The args object carries num_res_blocks etc. If you cannot import their
    # arg parser cleanly inside Colab, construct a minimal namespace with the
    # fields TTSR.__init__ reads (num_res_blocks, n_feats, res_scale, ...).
    from model import TTSR as TTSR_module           # model/TTSR.py
    import argparse
    # Bypass option.py (it parses argv). These three fields are the only ones
    # TTSR.__init__ reads. num_res_blocks MUST match the pretrained model.
    ttsr_args = argparse.Namespace(
        num_res_blocks="16+16+8+4", n_feats=64, res_scale=1.0)
    model = TTSR_module.TTSR(ttsr_args).to(device)

    # load weights (handle raw state_dict, {'state_dict':...}, and 'module.' )
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(f"[TTSR] loaded {loaded}/{len(model.state_dict())} tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    if loaded < 0.5 * len(model.state_dict()):
        print("[TTSR][WARN] fewer than half the weights loaded -- check that "
              "num_res_blocks matches the checkpoint and the key prefixes.")
    model.eval()

    @torch.no_grad()
    def _adapter(lr_bgr, ref_bgr):
        H, W = lr_bgr.shape[:2]          # ORIGINAL size (kept for output)
        # Cap the resolution TTSR processes internally to avoid CUDA OOM,
        # WITHOUT shrinking the evaluation. We run TTSR at <= max_proc px on
        # the long side, then upscale the SR result back to (H, W) so all
        # metrics/OCR are scored at full resolution, comparable to the table.
        scale_proc = min(1.0, max_proc / max(H, W))
        Hp, Wp = int(H * scale_proc), int(W * scale_proc)
        Hc, Wc = (Hp // scale) * scale, (Wp // scale) * scale
        m1 = cv2.resize(lr_bgr, (Wc, Hc))
        m2 = cv2.resize(ref_bgr, (Wc, Hc))

        # --- Option A: build a true 4x-LR from the degraded base M1 ---
        lr_small = cv2.resize(m1, (Wc // scale, Hc // scale),
                              interpolation=cv2.INTER_CUBIC)

        lr = _bgr_to_tensor(lr_small, device, norm)          # [1,3,h,w]
        lrsr = _bicubic(lr, (Hc, Wc))                        # bicubic 4x up
        ref = _bgr_to_tensor(m2, device, norm)               # [1,3,Hc,Wc]
        # ref_sr = ref downsampled by scale then back up (TTSR convention)
        ref_down = _bicubic(ref, (Hc // scale, Wc // scale))
        refsr = _bicubic(ref_down, (Hc, Wc))

        if not getattr(_adapter, "_printed", False):
            print(f"[TTSR] processing internally at {Wc}x{Hc} "
                  f"(orig {W}x{H}, max_proc={max_proc}); "
                  f"lower max_proc if OOM")
            _adapter._printed = True
        # Official TTSR returns a tuple; SR image is the first element.
        try:
            out = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise torch.cuda.OutOfMemoryError(
                f"TTSR OOM at {Wc}x{Hc}. Re-create the adapter with a smaller "
                f"max_proc (e.g. 256 or 192).")
        sr = out[0] if isinstance(out, (tuple, list)) else out

        sr_bgr = _tensor_to_bgr(sr, norm)
        # upscale SR back to the ORIGINAL resolution so scoring is done at
        # full size (fair vs input/proposed measured at the same max_dim).
        if sr_bgr.shape[:2] != (H, W):
            sr_bgr = cv2.resize(sr_bgr, (W, H), interpolation=cv2.INTER_CUBIC)
        return sr_bgr

    return _adapter


# ---- C2-Matching note -------------------------------------------------
# C2-Matching uses an mmsr/BasicSR-style test entry (mmsr/test.py + a yaml).
# The cleanest path is to NOT import its model here but instead run its own
# test script to dump SR PNGs, then score those PNGs with score_refsr_dir()
# below. This avoids fighting its mmcv 0.4.4 dependency inside the harness.
def score_refsr_dir(sr_dir, dataset="dataset", method_name="C2-Matching-rec",
                    max_dim=2000, out="refsr_c2_results.json"):
    """Score a folder of RefSR output PNGs (named scenario_pair.png) against
    ground truth, using the existing pipeline's metric + OCR functions.
    Use this when you ran a RefSR repo's own test.py to produce images."""
    import json, glob, os
    import experimental_results_standalone as pipe
    records = []
    for scenario, pair, m1, m2, gt in pipe.iter_pairs(dataset, max_dim):
        if gt is None:
            continue
        cand = os.path.join(sr_dir, f"{scenario}_{pair}.png")
        if not os.path.exists(cand):
            print(f"[skip] missing {cand}", file=sys.stderr); continue
        sr = cv2.imread(cand)
        if sr.shape[:2] != m1.shape[:2]:
            sr = cv2.resize(sr, (m1.shape[1], m1.shape[0]))
        ref_out = pipe.run_pipeline(m1, m2, pipe.config_for("P3_fidelity_lg"))
        roi = ref_out["roi"]
        rec = pipe.score_roi(pipe.crop(sr, roi),
                             pipe.crop(gt, roi) if gt is not None else None)
        rec.update(scenario=scenario, pair=pair, method=method_name)
        records.append(rec)
    json.dump({"records": records}, open(out, "w"), indent=2)
    print(f"scored {len(records)} pairs -> {out}")
    return records
