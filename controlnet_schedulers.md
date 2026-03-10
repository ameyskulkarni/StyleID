# ControlNet Per-Layer Timestep Scheduling

## Overview

Extended the ControlNet integration to support **per-layer, per-timestep conditioning scales** via two compact parameter lists, while preserving full backward compatibility with the original uniform scalar (`--cn_scale`).

---

## Changes

### `controlnet_utils.py` — `ControlNetWrapper.get_residuals`

**Added parameter:** `layer_scales: list[float] | None = None`

When `layer_scales` is provided:
- The diffusers `ControlNetModel` is called with `conditioning_scale=1.0` (unscaled residuals).
- Each down-block residual `i` is multiplied by `layer_scales[i]`.
- The mid-block residual is multiplied by `layer_scales[N]`, where `N = len(down_block_res_samples)`.

When `layer_scales` is `None` (default): behavior is identical to before — `conditioning_scale` is passed directly to the diffusers model.

---

### `controlnet_ddim_patch.py` — `patch_ddim_for_controlnet`

**Added parameters:**
- `start_scale_per_layer: list[float] | None` — per-layer scales at the first denoising timestep (t=0).
- `end_scale_per_layer: list[float] | None` — per-layer scales at the last denoising timestep (t=T-1).

Both are stored on the sampler as `_cn_start_scale` and `_cn_end_scale`.

**Added helper:** `_compute_layer_scales(sampler, index)` — called at every DDIM step to compute the effective per-layer scale vector for the current timestep index.

DDIM `index` counts down from `S-1` (first denoising step) to `0` (last step). The helper maps this to a forward step counter `t = S-1-index` ∈ [0, S-1], then linearly interpolates:

```
effective_scale[i] = start[i] + (end[i] - start[i]) * (t / (T - 1))
```

The resulting `layer_scales` list is forwarded to `get_residuals`.

**Three behavioral modes** (determined at each step by which lists are set):

| `start_scale_per_layer` | `end_scale_per_layer` | Behavior |
|:---|:---|:---|
| `None` | — | Uniform scalar `cn_scale` across all layers and timesteps (original behavior). |
| provided | `None` | Fixed per-layer scale, constant across all timesteps. |
| provided | provided | Linearly interpolated per-layer scale across timesteps. |

---

### `run_styleid.py`

**Added CLI arguments:**

```
--cn_start_scale_per_layer  Comma-separated floats, one per ControlNet residual layer
                             (13 values for SD 1.5: 12 down-block layers + 1 mid-block).
                             Overrides --cn_scale when set.

--cn_end_scale_per_layer    Comma-separated floats, same length as --cn_start_scale_per_layer.
                             When omitted, start scales are held constant across timesteps.
```

**Added parsing block** (immediately after `parser.parse_args()`): converts the comma-separated strings to `opt.cn_start_scale_list` and `opt.cn_end_scale_list`, with a length mismatch check.

**Updated `patch_ddim_for_controlnet` call** to pass the two new lists.

---

## Usage Examples

```bash
# Original behavior (unchanged): uniform scale across all layers and timesteps
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth --cn_scale 1.0

# Fixed per-layer scale (no timestep interpolation):
# stronger on early encoder layers, weaker at mid-block
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth \
  --cn_start_scale_per_layer 0.5,0.5,0.5,0.5,0.8,0.8,0.8,0.8,1.0,1.0,1.0,1.0,1.0

# Ramp down from strong → weak across timesteps, uniform across layers
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth \
  --cn_start_scale_per_layer 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 \
  --cn_end_scale_per_layer   0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3

# Full schedule: per-layer AND per-timestep
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth \
  --cn_start_scale_per_layer 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 \
  --cn_end_scale_per_layer   0.0,0.0,0.0,0.0,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5
```

---

## Layer Index Reference (SD 1.4/1.5)

Each entry in `cn_start_scale_per_layer` / `cn_end_scale_per_layer` corresponds to a ControlNet residual
that is added to an encoder skip connection, which is then consumed by a specific decoder output block.
The decoder pops skip connections in reverse order (`hs.pop()`), so the index-to-decoder mapping is inverted.

| `layer_scales` index | Encoder block | Decoder `output_block` | StyleID gamma layer? |
|:---|:---|:---|:---|
| **0** | input_block 0 | output_block 11 | **yes** |
| **1** | input_block 1 | output_block 10 | **yes** |
| **2** | input_block 2 | output_block 9 | **yes** |
| **3** | input_block 3 | output_block 8 | **yes** |
| **4** | input_block 4 | output_block 7 | **yes** |
| **5** | input_block 5 | output_block 6 | **yes** |
| 6 | input_block 6 | output_block 5 | no |
| 7 | input_block 7 | output_block 4 | no |
| 8 | input_block 8 | output_block 3 | no |
| 9 | input_block 9 | output_block 2 | no |
| 10 | input_block 10 | output_block 1 | no |
| 11 | input_block 11 | output_block 0 | no |
| 12 | mid_block | (all decoder blocks) | no |

### Targeting StyleID gamma layers only (output_blocks 6–11)

Modify **indices 0–5**; set indices 6–12 to your desired default.
Within those six indices, the correspondence to StyleID's gamma layer numbering is reversed:

| `layer_scales` index | StyleID gamma layer |
|:---|:---|
| 0 | layer 11 |
| 1 | layer 10 |
| 2 | layer 9 |
| 3 | layer 8 |
| 4 | layer 7 |
| 5 | layer 6 |

Example — ramp down ControlNet across timesteps for gamma layers only, silence all others:

```bash
python run_styleid.py --cnt data/cnt --sty data/sty --controlnet depth \
  --cn_start_scale_per_layer 1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 \
  --cn_end_scale_per_layer   0.3,0.3,0.3,0.3,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

---

## Files Changed

| File | Change |
|:---|:---|
| `controlnet_utils.py` | `get_residuals`: added `layer_scales` parameter, manual per-layer multiplication |
| `controlnet_ddim_patch.py` | `patch_ddim_for_controlnet`: added `start/end_scale_per_layer`; added `_compute_layer_scales` helper; updated `get_residuals` call |
| `run_styleid.py` | Added `--cn_start_scale_per_layer` / `--cn_end_scale_per_layer` args, parsing, and wiring |
