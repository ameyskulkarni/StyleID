# StyleID + ControlNet Integration

## Overview

This integrates pretrained ControlNet models into StyleID's style transfer pipeline to provide structural guidance (depth, edges, etc.) during the denoising pass, improving content structure preservation in stylized outputs.

**No training required** — uses pretrained ControlNet weights from HuggingFace.

## Files

Place these files in the **root** of the StyleID repository (same directory as the original `run_styleid.py`):

```
StyleID/
├── run_styleid.py              ← REPLACE with the new version
├── controlnet_utils.py         ← NEW: ControlNet loading + condition extraction
├── controlnet_patch.py         ← NEW: UNet monkey-patches for ControlNet residuals
├── controlnet_ddim_patch.py    ← NEW: DDIM sampler patch for ControlNet
├── ldm/                        ← unchanged
├── evaluation/                 ← unchanged
└── ...
```

## Installation

Additional dependencies (on top of existing StyleID environment):

```bash
pip install diffusers transformers accelerate
```

The ControlNet weights (~1.4 GB) will be downloaded automatically from HuggingFace on first run.

## Usage

### Original StyleID (no ControlNet, unchanged behavior):
```bash
python run_styleid.py --cnt data/cnt --sty data/sty
```

### StyleID + ControlNet (depth):
```bash
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth
```

### Adjusting ControlNet strength:
```bash
# Stronger structural guidance
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --cn_scale 1.5

# Weaker (more style freedom)
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --cn_scale 0.5
```

### Save depth maps for inspection:
```bash
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --save_condition
# Depth maps saved to output/conditions/
```

### Use ControlNet v1.1 (improved models):
```bash
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --cn_version 1.1
```

### Custom ControlNet model:
```bash
python run_styleid.py --cnt data/cnt --sty data/sty \
  --controlnet depth --cn_model "lllyasviel/control_v11f1p_sd15_depth"
```

## New CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--controlnet` | None | Condition type: `depth`, `canny`, `normal`, `seg`, `hed` |
| `--cn_scale` | 1.0 | ControlNet conditioning scale (0.0–2.0) |
| `--cn_model` | auto | Custom HuggingFace model ID |
| `--cn_version` | 1.0 | ControlNet version (1.0 or 1.1) |
| `--save_condition` | False | Save extracted condition maps |

## Evaluation

Evaluation commands are **unchanged**:

```bash
# First, copy inputs for evaluation
python util/copy_inputs.py --cnt data/cnt --sty data/sty

# Run StyleID + ControlNet to generate outputs
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --cn_scale 1.0

# Evaluate
cd evaluation
python eval_artfid.py --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```

## How It Works

ControlNet adds structural conditioning to StyleID's denoising process:

1. **DDIM Inversion** (content & style): Unchanged — extracts attention features normally
2. **Condition extraction**: Content image → depth map via MiDaS
3. **Denoising (reverse) pass**: At each timestep, ControlNet processes the noisy latent + depth map → produces residuals → added to UNet skip connections and middle block → UNet also receives StyleID's injected attention features (q/k/v) as before

The ControlNet residuals guide the UNet to preserve spatial structure from the depth map, while StyleID's attention injection handles the style transfer.

## VRAM Considerations

### Current situation
- StyleID alone: ~21 GB VRAM
- ControlNet model: ~1.4 GB additional (in fp16)
- Depth estimator (MiDaS): ~1 GB (loaded temporarily, freed after extraction)
- **Total peak**: ~23–24 GB

### Strategies for your RTX 3090 (24 GB):

1. **CPU offloading for ControlNet** (recommended first try):
   The depth estimator is already freed after extraction. If still OOM, add to `controlnet_utils.py`:
   ```python
   # After extracting condition, before denoising:
   cn_wrapper.to_cpu()  # move ControlNet to CPU
   # ... run denoising (ControlNet residuals computed per-step on CPU then moved to GPU)
   cn_wrapper.to_device()
   ```

2. **Use xformers memory-efficient attention**:
   ```bash
   pip install xformers
   ```
   Then uncomment the xformers line in the SD loading code.

3. **Smaller SD model**: StyleID uses SD 1.4, which is already the smallest standard SD (the UNet is ~860M params). There is no significantly smaller SD variant with ControlNet support. However, you can:
   - Use `torch.float16` precision (change `--precision full` to `autocast` which is already the default)
   - Reduce `--ddim_inv_steps` and `--save_feat_steps` from 50 to 30–20 (trades quality for memory)

4. **Disable precomputed features** to reduce peak memory from pickle loading:
   ```bash
   python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --precomputed ""
   ```

5. **Sequential ControlNet**: Compute ControlNet residuals for all timesteps first, cache them, then move ControlNet to CPU before running the full denoising. This adds a function in `controlnet_ddim_patch.py` (future improvement).

### Can you use a smaller model?
SD 1.4/1.5 is already the smallest model that ControlNet has pretrained weights for. The alternatives:
- **SD Tiny/Distilled**: No ControlNet weights exist
- **SD 2.x**: Same or larger UNet, no VRAM savings
- **SDXL**: Much larger, not suitable

So SD 1.4 + ControlNet SD 1.5 is the most VRAM-efficient option with pretrained ControlNet.

---

## Answers to Your Questions

### 1. Can you go phase-wise?
Yes. The code is structured for this. Currently only `depth` is implemented. To add a new condition type (e.g., canny), you just need to implement the `_extract_canny()` function in `controlnet_utils.py` (canny is already implemented as an example — it doesn't need a GPU, just OpenCV). Then run with `--controlnet canny`.

### 2. Single vs separate ControlNets? Multi-input?
Each condition type (depth, canny, normal, seg, etc.) has its **own separately trained ControlNet**. There is no single ControlNet that handles all input types.

For **multi-condition** control, the standard approach is to load multiple ControlNets and **sum their output residuals** before adding to the UNet. This is straightforward to add:
```python
# Pseudocode for multi-ControlNet (future phase):
residuals_depth = cn_depth.get_residuals(...)
residuals_canny = cn_canny.get_residuals(...)
combined_down = [d1 + d2 for d1, d2 in zip(residuals_depth[0], residuals_canny[0])]
combined_mid = residuals_depth[1] + residuals_canny[1]
```
However, multi-ControlNet will use ~2.8 GB more VRAM (one extra ControlNet).

### 3. Running after integration
See [Usage](#usage) above. The entry point and evaluation commands are unchanged.

---

## Research Ideas (CVPR 2026 Workshop, 4–6 week timeline)

### Proposed Title
"Structure-Guided Style Transfer via ControlNet-Enhanced Attention Injection in Diffusion Models"

### Core Contribution
Combining training-free attention-based style injection (StyleID) with structural conditioning (ControlNet) to improve content preservation without sacrificing style fidelity. This is a **simple, novel, and reproducible** combination not explored in existing work.

### Experiments

**Experiment 1: Quantitative comparison (essential)**
- Baseline: StyleID (original)
- Ours: StyleID + ControlNet-Depth at various scales (0.25, 0.5, 0.75, 1.0, 1.5)
- Metrics: ArtFID, FID (style), LPIPS (content), CFSD
- Dataset: Same as StyleID paper (20 content from MS-COCO, 40 style from WikiArt → 800 pairs)
- **Expected result**: Lower LPIPS and CFSD (better content preservation) with comparable FID (similar style quality). ArtFID should improve since content fidelity improves.

**Experiment 2: Ablation across condition types**
- Compare: depth vs canny vs normal vs seg as ControlNet conditions
- Same metrics, same dataset
- Shows which structural signal best preserves content for style transfer
- This is simple to run once you implement the other extractors

**Experiment 3: cn_scale sweep (Pareto curves)**
- Plot LPIPS (content) vs FID (style) as cn_scale varies from 0 to 2
- Compare against StyleID's gamma sweep
- Shows the new controllability dimension ControlNet adds

**Experiment 4: Qualitative comparison grid**
- 5 content × 5 style × {StyleID, StyleID+Depth, StyleID+Canny}
- Visual figure showing structural improvement

**Experiment 5 (optional, if time permits): Multi-ControlNet**
- Combine depth + canny ControlNets
- Show further improvement in content preservation

### Related Work Positioning
- StyleID (CVPR 2024): Training-free style transfer via attention injection
- ControlNet (ICCV 2023): Structural conditioning for diffusion
- InstantStyle-Plus (2024): Uses Tile ControlNet for content preservation, but requires IP-Adapter and is a different pipeline
- ICAS (2025): IP-Adapter + ControlNet, but requires fine-tuning
- **Our gap**: No work combines StyleID's attention injection with ControlNet in a fully training-free manner. We show that depth/edge guidance from ControlNet complements attention-based style injection.

### Timeline (4–6 weeks)
- Week 1: Get baseline running, implement canny extractor, run Exp 1 & 3
- Week 2: Implement normal/seg extractors, run Exp 2
- Week 3: Run full quantitative evaluation, generate qualitative figures (Exp 4)
- Week 4: Write paper, create figures/tables
- Week 5: Polish, ablations, submit
