# StyleID: Complete Guide - Paper Theory to Code Implementation

## Table of Contents
1. [Paper Core Ideas & Contributions](#1-paper-core-ideas--contributions)
2. [Repository Structure](#2-repository-structure)
3. [Paper Concepts → Code Mapping](#3-paper-concepts--code-mapping)
4. [Key Modules & Implementation Details](#4-key-modules--implementation-details)
5. [Experimental Workflow Guide](#5-experimental-workflow-guide)
6. [Quick Reference](#6-quick-reference)

---

## 1. Paper Core Ideas & Contributions

### 1.1 The Main Problem
Existing diffusion-based style transfer methods either:
- **Require optimization** (fine-tuning, textual inversion) → time-consuming
- **Fail to leverage large-scale models** like Stable Diffusion effectively

### 1.2 The StyleID Solution: Training-Free Style Transfer

The paper proposes a **zero-shot, training-free** method that manipulates self-attention features during inference to achieve style transfer.

#### Core Innovation: Self-Attention Manipulation
Instead of treating self-attention (SA) as it's normally used (Q, K, V all from the same image), StyleID manipulates it like cross-attention:

**Normal Self-Attention:**
```
Q, K, V ← same image
Output = Attention(Q, K, V)
```

**StyleID's Style Injection:**
```
Q ← content image
K, V ← style image  
Output = Attention(Q, K_style, V_style)
```

### 1.3 Three Key Components

#### Component 1: **Attention-based Style Injection**
- **What:** Substitute Key and Value of content with those from style image
- **Where:** Applied to decoder layers 7-11 (the latter part handling local textures)
- **Why:** Transfers style textures to content while maintaining spatial layout through attention maps

**Key Equations:**
```
Q̃_t^cs = γ × Q_t^c + (1-γ) × Q_t^cs  (Query Preservation)
φ_out^cs = Attention(Q̃_t^cs, K_t^s, V_t^s)  (Style Injection)
```

#### Component 2: **Query Preservation**
- **Problem:** Pure K,V substitution disrupts content structure
- **Solution:** Blend content query (Q^c) with stylized query (Q^cs)
- **Control:** Parameter γ ∈ [0,1]
  - Higher γ (e.g., 0.75) → more content fidelity
  - Lower γ (e.g., 0.3) → stronger style transfer

#### Component 3: **Attention Temperature Scaling**
- **Problem:** K,V substitution causes attention maps to become blurred (lower std dev)
- **Solution:** Multiply attention logits by τ > 1 before softmax
- **Effect:** Sharpens attention maps, preserves details

**Equation:**
```
Attention_τ(Q̃, K, V) = softmax(τ × Q̃K^T / √d) · V
```
Default: τ = 1.5

#### Component 4: **Initial Latent AdaIN**
- **Problem:** Style injection alone fails to transfer colors properly
- **Solution:** Modulate the initial noise statistics
- **Method:** Apply AdaIN to align content noise with style noise statistics

**Equation:**
```
z_T^cs = σ(z_T^s) × (z_T^c - μ(z_T^c)) / σ(z_T^c) + μ(z_T^s)
```

### 1.4 Key Characteristics & Advantages

1. **Training-free:** No optimization, works with pre-trained Stable Diffusion
2. **Texture-aware:** Transfers similar styles to patches with similar local textures
3. **Content-preserving:** Maintains spatial structure through attention mechanism
4. **Controllable:** Adjustable style-content trade-off via γ parameter

---

## 2. Repository Structure

### 2.1 Directory Organization

```
StyleID/
├── diffusers_implementation/          # Modern implementation (Recommended)
│   ├── run_styleid_diffusers.py      # Main script (diffusers-based)
│   ├── stable_diffusion.py            # SD utilities (load, encode, decode)
│   ├── config.py                      # Argument parser
│   └── utils.py                       # Image I/O utilities
│
├── run_styleid.py                     # Original implementation script
│
├── ldm/                               # LDM/Stable Diffusion components
│   ├── models/
│   │   ├── diffusion/
│   │   │   ├── ddim.py               # DDIM sampler (CRITICAL for inversion)
│   │   │   └── ddpm.py               # Base diffusion model
│   │   └── autoencoder.py            # VAE encoder/decoder
│   └── modules/
│       ├── attention.py               # Attention mechanisms
│       └── diffusionmodules/
│           └── openaimodel.py         # UNet architecture
│
├── evaluation/                        # Evaluation metrics
│   ├── eval_artfid.py                # Art-FID evaluation
│   └── eval_histogan.py              # Histogram loss evaluation
│
├── data/                              # Sample images
│   ├── cnt/                          # Content images
│   └── sty/                          # Style images
│
└── models/                           # Pre-trained model weights
    └── ldm/stable-diffusion-v1/
```

### 2.2 Two Implementations Available

#### **Option 1: diffusers_implementation/** (Recommended for experiments)
- Modern, clean code using HuggingFace diffusers
- Easier to understand and modify
- Supports SD 1.5, 2.0, 2.1
- Better for learning and experimentation

#### **Option 2: Original Implementation**
- Based on CompVis/stable-diffusion codebase
- Used for paper's quantitative results
- More complex but faithful to original experiments

---

## 3. Paper Concepts → Code Mapping

### 3.1 Attention-based Style Injection

**Paper Concept (Section 4.1):**
> "We substitute the key and value of content image with those of style for transferring the texture of style image into the content image."

**Code Location:** `diffusers_implementation/run_styleid_diffusers.py`

```python
class style_transfer_module():
    def __init__(self, ...):
        # Lines 31-39: Register hooks on attention layers
        self.injection_layers = [7, 8, 9, 10, 11]  # Decoder layers
        
    def register_attn_hooks(self, attn_layer, name, mode):
        # Lines 100-130: Hook functions to capture and inject features
        if mode == 'get':
            # Capture K, V from style during inversion
            hook = lambda module, input, output: self.get_attn_features(...)
        elif mode == 'modify':
            # Inject K, V into content during generation
            hook = lambda module, input, output: self.modify_attn_features(...)
```

**Key Implementation Details:**
- **Line 118-120:** Save style's K, V during DDIM inversion
- **Line 158-163:** Inject saved K, V during reverse diffusion
- **Line 146-152:** Apply query preservation (γ blending)

### 3.2 Query Preservation

**Paper Equation (3):**
```
Q̃_t^cs = γ × Q_t^c + (1-γ) × Q_t^cs
```

**Code Implementation:** `diffusers_implementation/run_styleid_diffusers.py`

```python
def modify_attn_features(self, name, attn_output, is_cross):
    # Line 146-152
    gamma = self.style_transfer_params['gamma']  # Default: 0.75
    
    # Blend queries
    query_modified = gamma * self.attn_features['query_content'][name] + \
                     (1 - gamma) * query_stylized
```

**Configuration:** `config.py`
```python
parse.add_argument('--gamma', type=float, default=0.75)
# Usage: --gamma 0.75  (more content)
#        --gamma 0.3   (more style)
```

### 3.3 Attention Temperature Scaling

**Paper Equation (5):**
```
Attention_τ(Q̃, K, V) = softmax(τ × Q̃K^T / √d) · V, τ > 1
```

**Code Implementation:** `diffusers_implementation/stable_diffusion.py`

```python
def attention_op(attn, hidden_states, ..., temperature=1.0):
    # Line 107-115: Compute Q, K, V
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    
    # Line 118-119: Apply temperature scaling
    query = query * (temperature ** 0.5)  # Scale query before attention
    
    # Line 121-124: Compute attention
    attention_scores = attn.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
```

**Configuration:**
```python
parse.add_argument('--T', type=float, default=1.5)
# Default τ = 1.5 (average ratio across timesteps)
```

### 3.4 Initial Latent AdaIN

**Paper Equation (6):**
```
z_T^cs = σ(z_T^s) × (z_T^c - μ(z_T^c)) / σ(z_T^c) + μ(z_T^s)
```

**Code Implementation:** `diffusers_implementation/run_styleid_diffusers.py`

```python
def initial_latent_adain(self, latent_content, latent_style):
    # Line 230-240: AdaIN on initial noise
    mean_content = latent_content.mean(dim=(2,3), keepdim=True)
    std_content = latent_content.std(dim=(2,3), keepdim=True)
    
    mean_style = latent_style.mean(dim=(2,3), keepdim=True)
    std_style = latent_style.std(dim=(2,3), keepdim=True)
    
    # Normalize and transfer statistics
    normalized = (latent_content - mean_content) / (std_content + 1e-5)
    stylized_latent = normalized * std_style + mean_style
    
    return stylized_latent
```

**Configuration:**
```python
parse.add_argument('--without_init_adain', action='store_true')
# By default, AdaIN is applied unless this flag is set
```

### 3.5 DDIM Inversion

**Paper (Section 4.1):**
> "We first obtain the latent for content and style images with DDIM inversion, and then collect the SA features of style image over the DDIM inversion process."

**Code Implementation:** `diffusers_implementation/run_styleid_diffusers.py`

```python
def invert_process(self, input, denoise_kwargs):
    # Line 290-320: Reverse timesteps for inversion
    timesteps = reversed(self.scheduler.timesteps)
    
    for i in tqdm(range(0, num_inference_steps)):
        t = timesteps[i]
        
        # Forward diffusion step (adding predicted noise)
        noise_pred = self.unet(cur_latent, t, **denoise_kwargs).sample
        cur_latent = self.scheduler.step(
            noise_pred, t, cur_latent, 
            reverse_process=True  # Key: running in reverse
        ).prev_sample
```

**Two inversions happen:**
1. Content image: `z_0^c → z_T^c` (save Q features)
2. Style image: `z_0^s → z_T^s` (save K, V features)

---

## 4. Key Modules & Implementation Details

### 4.1 Main Execution Flow

**File:** `diffusers_implementation/run_styleid_diffusers.py`

```python
def main():
    # 1. Load Stable Diffusion components
    unet, vae, text_encoder, tokenizer, scheduler = load_stable_diffusion(cfg)
    
    # 2. Load content and style images
    cnt_img = Image.open(cfg.cnt_fn).resize((512, 512))
    sty_img = Image.open(cfg.sty_fn).resize((512, 512))
    
    # 3. Encode to latent space
    cnt_latent = encode_latent(cnt_img, vae)
    sty_latent = encode_latent(sty_img, vae)
    
    # 4. Create style transfer module
    style_module = style_transfer_module(
        unet, vae, text_encoder, tokenizer, scheduler, cfg,
        style_transfer_params={
            'gamma': cfg.gamma,
            'tau': cfg.T,
            'injection_layers': cfg.layers
        }
    )
    
    # 5. DDIM Inversion (extract features)
    # 5a. Invert content (save Q)
    style_module.set_mode('get_query')
    cnt_noise = style_module.invert_process(cnt_latent, denoise_kwargs)
    
    # 5b. Invert style (save K, V)
    style_module.set_mode('get_kv')
    sty_noise = style_module.invert_process(sty_latent, denoise_kwargs)
    
    # 6. Initial Latent AdaIN
    if not cfg.without_init_adain:
        init_noise = style_module.initial_latent_adain(cnt_noise, sty_noise)
    else:
        init_noise = cnt_noise
    
    # 7. Style Transfer (reverse diffusion with injection)
    style_module.set_mode('inject')
    stylized_latent = style_module.denoise_process(
        init_noise, denoise_kwargs, temperature=cfg.T
    )
    
    # 8. Decode to image
    stylized_img = decode_latent(stylized_latent, vae)
    
    # 9. Save result
    stylized_img.save(f'{cfg.save_dir}/stylized.png')
```

### 4.2 Critical Classes & Their Roles

#### **Class: `style_transfer_module`**
**Purpose:** Orchestrates the entire style transfer pipeline

**Key Methods:**

1. **`__init__(self, ...)`**
   - Sets up attention hooks on specified layers
   - Initializes feature storage dictionaries
   
2. **`register_attn_hooks(self, attn_layer, name, mode)`**
   - Registers forward hooks on attention layers
   - Mode: 'get' (capture features) or 'modify' (inject features)
   
3. **`get_attn_features(self, name, attn_output, is_cross)`**
   - Called during inversion
   - Saves Q (content), K, V (style) to `self.attn_features`
   
4. **`modify_attn_features(self, name, attn_output, is_cross)`**
   - Called during generation
   - Retrieves saved features
   - Applies query preservation
   - Returns modified attention output
   
5. **`initial_latent_adain(self, latent_content, latent_style)`**
   - Transfers color statistics from style to content noise
   
6. **`invert_process(self, input, denoise_kwargs)`**
   - Performs DDIM inversion
   - Runs diffusion in reverse (image → noise)
   
7. **`denoise_process(self, init_noise, denoise_kwargs)`**
   - Performs reverse diffusion (noise → image)
   - Applies style injection at each timestep

### 4.3 Attention Hook Mechanism

**The Hook System:**
PyTorch forward hooks intercept layer outputs and modify them on-the-fly.

```python
# Registration (Line 75-85)
def register_attn_hooks(self, attn_layer, name, mode):
    if mode == 'get':
        hook = lambda module, input, output: self.get_attn_features(name, output, ...)
    elif mode == 'modify':
        hook = lambda module, input, output: self.modify_attn_features(name, output, ...)
    
    handle = attn_layer.register_forward_hook(hook)
    self.hook_handles.append(handle)

# Cleanup
def remove_hooks(self):
    for handle in self.hook_handles:
        handle.remove()
```

**Flow:**
```
1. Forward pass begins
2. Layer computation
3. Hook intercepts output
4. Hook function processes/modifies
5. Modified output continues forward
```

### 4.4 Stable Diffusion Utilities

**File:** `diffusers_implementation/stable_diffusion.py`

**Key Functions:**

```python
def load_stable_diffusion(cfg):
    """Load SD components based on version"""
    if cfg.sd_version == "1.5":
        model_id = "runwayml/stable-diffusion-v1-5"
    elif cfg.sd_version == "2.1-base":
        model_id = "stabilityai/stable-diffusion-2-1-base"
    # ... load UNet, VAE, text encoder, scheduler
    return unet, vae, text_encoder, tokenizer, scheduler

def encode_latent(image, vae):
    """Encode PIL image to latent space"""
    # Convert to tensor, normalize [-1, 1]
    # VAE encode: 3×512×512 → 4×64×64
    return latent

def decode_latent(latent, vae):
    """Decode latent to PIL image"""
    # VAE decode: 4×64×64 → 3×512×512
    # Denormalize, convert to PIL
    return image

def get_unet_layers(unet):
    """Extract decoder ResNet and Attention layers"""
    # Returns lists of layers 0-11
    return resnet_layers, attn_layers

def attention_op(attn, hidden_states, ..., temperature=1.0):
    """Custom attention operation with temperature scaling"""
    # Computes attention with optional temperature
    return output
```

---

## 5. Experimental Workflow Guide

### 5.1 Quick Start (Diffusers Implementation)

```bash
# Install dependencies
pip install torch torchvision diffusers transformers accelerate

# Basic usage (default settings)
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn data/cnt.png \
    --sty_fn data/sty.png \
    --sd_version 2.1-base

# High style fidelity
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn data/cnt.png \
    --sty_fn data/sty.png \
    --gamma 0.3 \
    --T 1.5

# High content fidelity
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn data/cnt.png \
    --sty_fn data/sty.png \
    --gamma 0.9 \
    --T 1.5
```

### 5.2 Parameter Tuning Guide

#### **γ (gamma) - Style-Content Trade-off**
```
Value Range: [0.0, 1.0]
Default: 0.75

γ = 0.0  → Pure style (content lost)
γ = 0.3  → Strong style transfer
γ = 0.5  → Balanced
γ = 0.75 → Default (good balance)
γ = 0.9  → Strong content preservation
γ = 1.0  → No style transfer
```

**Usage:**
```bash
--gamma 0.3   # For artistic/creative styles
--gamma 0.75  # For balanced results (default)
--gamma 0.9   # For subtle style hints
```

#### **τ (T) - Attention Temperature**
```
Value Range: [1.0, 2.0]
Default: 1.5

τ = 1.0  → No scaling (may be blurry)
τ = 1.5  → Default (sharp, well-defined)
τ = 2.0  → Very sharp (may over-sharpen)
```

**Usage:**
```bash
--T 1.0   # If results are too sharp/artifacts
--T 1.5   # Default setting (recommended)
--T 2.0   # If results are too blurry
```

#### **Injection Layers**
```
Default: [7, 8, 9, 10, 11] (layers handling local textures)

Early layers (0-3): Global structure/composition
Middle layers (4-6): Object shapes
Late layers (7-11): Textures and details
```

**Usage:**
```bash
--layers 7 8 9 10 11     # Default (local textures)
--layers 4 5 6 7 8 9 10 11  # More aggressive style
--layers 9 10 11         # Subtle texture transfer
```

#### **Initial Latent AdaIN**
```bash
# With AdaIN (default) - better color transfer
python run_styleid_diffusers.py --cnt_fn ... --sty_fn ...

# Without AdaIN - may preserve content colors
python run_styleid_diffusers.py --cnt_fn ... --sty_fn ... --without_init_adain
```

### 5.3 Ablation Studies

To understand each component's contribution:

```bash
# 1. Baseline (no style transfer)
python run_styleid_diffusers.py --without_attn_injection --without_init_adain

# 2. Only attention injection
python run_styleid_diffusers.py --without_init_adain

# 3. Only initial AdaIN (no injection)
python run_styleid_diffusers.py --without_attn_injection

# 4. Full method (all components)
python run_styleid_diffusers.py  # Default
```

### 5.4 Troubleshooting Common Issues

#### **Issue 1: Content is completely lost**
**Cause:** γ too low
**Solution:**
```bash
--gamma 0.75  # Increase to 0.8 or 0.9
```

#### **Issue 2: Style doesn't transfer well**
**Cause:** γ too high
**Solution:**
```bash
--gamma 0.3  # Decrease to 0.2-0.5
```

#### **Issue 3: Results are blurry**
**Cause:** Attention temperature too low
**Solution:**
```bash
--T 1.8  # Increase temperature to 1.8-2.0
```

#### **Issue 4: Colors don't match style**
**Cause:** Initial AdaIN disabled or not working
**Solution:**
```bash
# Ensure AdaIN is enabled (don't use --without_init_adain)
# Check if initial noise statistics are being transferred
```

#### **Issue 5: Artifacts or oversaturation**
**Possible causes:**
1. Temperature too high → `--T 1.2`
2. Too many injection layers → `--layers 9 10 11`
3. Style image has extreme colors → try normalizing style image

### 5.5 Advanced Experimentation

#### **Custom Attention Layer Selection**
Experiment with different layer combinations:

```python
# In run_styleid_diffusers.py, modify:
style_transfer_params = {
    'gamma': 0.75,
    'tau': 1.5,
    'injection_layers': [6, 7, 8]  # Try different combinations
}
```

**Layer characteristics:**
- **0-3:** Global layout, composition
- **4-6:** Object boundaries, shapes
- **7-9:** Fine textures, patterns
- **10-11:** Very fine details, brush strokes

#### **Time-dependent Parameter Scheduling**
Modify gamma or temperature across timesteps:

```python
def get_gamma_schedule(timestep, total_steps):
    """Example: Increase content preservation over time"""
    progress = timestep / total_steps
    return 0.3 + 0.5 * progress  # γ: 0.3 → 0.8
```

#### **Multi-style Transfer**
Blend multiple styles:

```python
# Collect features from multiple style images
style_features_1 = invert_and_save_features(style_img_1)
style_features_2 = invert_and_save_features(style_img_2)

# Blend K, V features
blended_K = 0.5 * style_features_1['K'] + 0.5 * style_features_2['K']
blended_V = 0.5 * style_features_1['V'] + 0.5 * style_features_2['V']

# Use blended features for injection
```

---

## 6. Quick Reference

### 6.1 Command Cheat Sheet

```bash
# Default balanced transfer
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style>

# Strong style
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --gamma 0.3

# Preserve content
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --gamma 0.9

# Sharper results
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --T 1.8

# Subtle texture only
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --layers 10 11

# No color transfer
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --without_init_adain

# Ablation: no style injection
python run_styleid_diffusers.py --cnt_fn <content> --sty_fn <style> --without_attn_injection
```

### 6.2 Key Files Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `diffusers_implementation/run_styleid_diffusers.py` | Main script | `style_transfer_module`, `main()` |
| `diffusers_implementation/stable_diffusion.py` | SD utilities | `load_stable_diffusion()`, `encode_latent()`, `attention_op()` |
| `diffusers_implementation/config.py` | Arguments | `get_args()` |
| `ldm/models/diffusion/ddim.py` | DDIM sampler | `DDIMSampler.encode_ddim()`, `p_sample_ddim()` |
| `ldm/modules/attention.py` | Attention layers | `Attention.forward()` |

### 6.3 Paper Section → Code Mapping

| Paper Section | Concept | Code Location |
|---------------|---------|---------------|
| 4.1 | Attention-based Style Injection | `run_styleid_diffusers.py:100-130` |
| 4.1 | Query Preservation | `run_styleid_diffusers.py:146-152` |
| 4.2 | Attention Temperature Scaling | `stable_diffusion.py:118-119` |
| 4.3 | Initial Latent AdaIN | `run_styleid_diffusers.py:230-240` |
| 3 (Method) | DDIM Inversion | `run_styleid_diffusers.py:290-320` |

### 6.4 Default Hyperparameters

```python
DEFAULTS = {
    'gamma': 0.75,           # Query preservation
    'tau': 1.5,             # Attention temperature
    'injection_layers': [7, 8, 9, 10, 11],  # Decoder layers
    'ddim_steps': 20,       # Number of diffusion steps
    'sd_version': '2.1-base',  # Stable Diffusion version
    'with_init_adain': True,   # Enable initial AdaIN
    'with_attn_injection': True  # Enable style injection
}
```

### 6.5 Mathematical Formulas Summary

#### Attention-based Style Injection
```
Q̃_t^cs = γ × Q_t^c + (1-γ) × Q_t^cs
φ_out^cs = Attention(Q̃_t^cs, K_t^s, V_t^s)
```

#### Temperature Scaling
```
Attention_τ(Q̃, K, V) = softmax(τ × Q̃K^T / √d) · V
```

#### Initial Latent AdaIN
```
z_T^cs = σ(z_T^s) × (z_T^c - μ(z_T^c)) / σ(z_T^c) + μ(z_T^s)
```

where:
- μ(·): channel-wise mean
- σ(·): channel-wise standard deviation
- ^c: content
- ^s: style
- ^cs: stylized result

---

## Appendix: Understanding DDIM Inversion

DDIM inversion is crucial for StyleID. Here's why:

### Why Inversion?
1. **Feature Extraction:** Need to collect attention features at each timestep
2. **Deterministic:** DDIM is deterministic (η=0), so inversion is exact
3. **Efficient:** Can reuse inverted noise for multiple experiments

### How It Works
**Normal Generation (Denoising):**
```
z_T (noise) → z_{T-1} → ... → z_1 → z_0 (image)
```

**DDIM Inversion (Reverse):**
```
z_0 (image) → z_1 → ... → z_{T-1} → z_T (noise)
```

**The Math:**
```
# Forward (generation): remove noise
z_{t-1} = √(α_{t-1}) · pred_x0 + √(1 - α_{t-1} - σ_t²) · ε_θ(z_t)

# Backward (inversion): add predicted noise
z_{t+1} = √(α_{t+1}) · pred_x0 + √(1 - α_{t+1}) · ε_θ(z_t)
```

### In StyleID:
1. Invert content image → collect Q features
2. Invert style image → collect K, V features
3. Generate with injected features → style transfer!

---

## Conclusion

StyleID achieves impressive style transfer through:
1. **Attention manipulation:** Injecting style K, V into content generation
2. **Query preservation:** Maintaining content structure (γ blending)
3. **Temperature scaling:** Keeping details sharp (τ scaling)
4. **Initial AdaIN:** Transferring color tones

The repository provides a clean, modular implementation that makes it easy to:
- Understand each component
- Experiment with parameters
- Extend to new ideas

**Happy experimenting! 🎨**
