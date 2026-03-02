# StyleID Practical Experimentation Guide

## Quick Start Examples

### Example 1: Basic Style Transfer
```bash
# Portrait photo + Van Gogh painting style
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn examples/portrait.jpg \
    --sty_fn examples/starry_night.jpg \
    --sd_version 2.1-base \
    --gamma 0.75 \
    --T 1.5
```

**Expected Result:**
- Portrait maintains facial features
- Acquires swirling brushstroke patterns
- Colors shift toward blue/yellow tones of Starry Night

---

### Example 2: Landscape Transformation
```bash
# Modern photo + Monet impressionist style
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn examples/modern_landscape.jpg \
    --sty_fn examples/monet_waterlilies.jpg \
    --sd_version 2.1-base \
    --gamma 0.5 \
    --T 1.5
```

**Expected Result:**
- Landscape structure preserved
- Soft, impressionist brushwork
- Pastel color palette

---

### Example 3: Architecture + Abstract Style
```bash
# Building photo + abstract geometric art
python diffusers_implementation/run_styleid_diffusers.py \
    --cnt_fn examples/building.jpg \
    --sty_fn examples/mondrian.jpg \
    --sd_version 2.1-base \
    --gamma 0.3 \
    --T 1.5 \
    --layers 7 8 9 10 11
```

**Expected Result:**
- Building outlines maintained
- Geometric patterns and color blocks
- Strong stylistic transformation

---

## Common Use Cases & Settings

### Use Case 1: Photorealistic Portrait with Subtle Artistic Touch
**Goal:** Add artistic flair without losing photo realism

```bash
python run_styleid_diffusers.py \
    --cnt_fn photo.jpg \
    --sty_fn art_style.jpg \
    --gamma 0.9 \          # HIGH - preserve photo realism
    --T 1.5 \
    --layers 10 11         # Only fine texture layers
```

**When to use:**
- Professional portraits
- Product photography
- Maintaining facial recognition

**Trade-offs:**
- ✓ Content preserved
- ✗ Style is subtle
- ✓ Natural looking

---

### Use Case 2: Creative Artistic Transformation
**Goal:** Strong artistic interpretation, content as reference

```bash
python run_styleid_diffusers.py \
    --cnt_fn photo.jpg \
    --sty_fn artwork.jpg \
    --gamma 0.3 \          # LOW - embrace style
    --T 1.5 \
    --layers 6 7 8 9 10 11  # Include shape layers
```

**When to use:**
- Digital art projects
- Creative experiments
- Album covers, posters

**Trade-offs:**
- ✓ Strong style transfer
- ✗ Content may be altered
- ✓ Artistic interpretation

---

### Use Case 3: Texture Transfer (Maintain Colors)
**Goal:** Transfer brushwork/texture but keep original colors

```bash
python run_styleid_diffusers.py \
    --cnt_fn photo.jpg \
    --sty_fn texture_source.jpg \
    --gamma 0.75 \
    --T 1.5 \
    --without_init_adain   # DISABLE color transfer
```

**When to use:**
- Apply texture patterns
- Maintain brand colors
- Color-sensitive applications

**Trade-offs:**
- ✓ Original colors preserved
- ✗ Style colors not transferred
- ✓ Texture patterns applied

---

### Use Case 4: Color Palette Transfer
**Goal:** Change color scheme, minimal texture changes

```bash
python run_styleid_diffusers.py \
    --cnt_fn photo.jpg \
    --sty_fn color_reference.jpg \
    --gamma 0.9 \
    --T 1.5 \
    --layers 11            # Minimal injection
```

**When to use:**
- Color grading
- Mood adjustment
- Season transformations (summer→autumn)

**Trade-offs:**
- ✓ Color scheme transferred
- ✗ Limited texture transfer
- ✓ Structure maintained

---

## Parameter Combinations for Different Effects

### Preset 1: "Balanced" (Default)
```python
gamma = 0.75
tau = 1.5
layers = [7, 8, 9, 10, 11]
with_init_adain = True
```
**Use when:** First attempt, unsure of desired result

---

### Preset 2: "Strong Style"
```python
gamma = 0.3
tau = 1.8
layers = [5, 6, 7, 8, 9, 10, 11]
with_init_adain = True
```
**Use when:** Want dramatic transformation, artistic freedom

---

### Preset 3: "Content Preservation"
```python
gamma = 0.9
tau = 1.2
layers = [9, 10, 11]
with_init_adain = False
```
**Use when:** Must maintain content fidelity, subtle changes only

---

### Preset 4: "Texture Focus"
```python
gamma = 0.75
tau = 1.5
layers = [8, 9, 10]
with_init_adain = False
```
**Use when:** Want brushwork/texture without color changes

---

### Preset 5: "Color + Minimal Texture"
```python
gamma = 0.85
tau = 1.3
layers = [10, 11]
with_init_adain = True
```
**Use when:** Color palette transfer is priority

---

## Debugging & Problem Solving

### Problem 1: "Colors are wrong/oversaturated"

**Symptom:** Output has unnatural, overly vibrant colors

**Possible Causes:**
1. Style image has extreme colors
2. Initial AdaIN transferring unusual color distribution
3. Temperature too high causing artifacts

**Solutions to try:**

```bash
# Solution 1: Disable AdaIN
python run_styleid_diffusers.py ... --without_init_adain

# Solution 2: Reduce temperature
python run_styleid_diffusers.py ... --T 1.2

# Solution 3: Use different style reference
# Pre-process style image: normalize/adjust colors
```

---

### Problem 2: "Content structure is lost"

**Symptom:** Can't recognize original content in result

**Possible Causes:**
1. γ (gamma) too low
2. Too many injection layers
3. Style image has very different structure

**Solutions to try:**

```bash
# Solution 1: Increase gamma
python run_styleid_diffusers.py ... --gamma 0.85

# Solution 2: Reduce injection layers
python run_styleid_diffusers.py ... --layers 9 10 11

# Solution 3: Combine both
python run_styleid_diffusers.py ... --gamma 0.9 --layers 10 11
```

---

### Problem 3: "Style doesn't transfer much"

**Symptom:** Output looks almost identical to content

**Possible Causes:**
1. γ (gamma) too high
2. Too few injection layers
3. Style image not distinctive enough

**Solutions to try:**

```bash
# Solution 1: Decrease gamma
python run_styleid_diffusers.py ... --gamma 0.3

# Solution 2: Increase injection layers
python run_styleid_diffusers.py ... --layers 6 7 8 9 10 11

# Solution 3: Use more distinctive style
# Choose artwork with clear textures/patterns
```

---

### Problem 4: "Output is blurry/lacks detail"

**Symptom:** Result appears soft, lacks sharpness

**Possible Causes:**
1. Temperature (τ) too low
2. Attention maps too smoothed

**Solutions to try:**

```bash
# Solution 1: Increase temperature
python run_styleid_diffusers.py ... --T 1.8

# Solution 2: If still blurry, go higher
python run_styleid_diffusers.py ... --T 2.0

# Solution 3: Check if style image itself is sharp
```

---

### Problem 5: "Artifacts/distortions appear"

**Symptom:** Strange patterns, unrealistic elements

**Possible Causes:**
1. Temperature too high
2. Style and content too incompatible
3. Extreme parameter values

**Solutions to try:**

```bash
# Solution 1: Reduce temperature
python run_styleid_diffusers.py ... --T 1.2

# Solution 2: Return to balanced settings
python run_styleid_diffusers.py ... --gamma 0.75 --T 1.5

# Solution 3: Try different style reference
```

---

## Advanced Techniques

### Technique 1: Multi-Pass Style Transfer

Apply StyleID multiple times with different styles:

```bash
# Pass 1: Apply color palette
python run_styleid_diffusers.py \
    --cnt_fn original.jpg \
    --sty_fn color_ref.jpg \
    --gamma 0.85 \
    --layers 10 11 \
    --save_dir pass1/

# Pass 2: Apply texture on result
python run_styleid_diffusers.py \
    --cnt_fn pass1/stylized.jpg \
    --sty_fn texture_ref.jpg \
    --gamma 0.7 \
    --layers 8 9 10 \
    --without_init_adain \
    --save_dir final/
```

---

### Technique 2: Partial Style Transfer

Apply style to specific regions using masks (requires code modification):

```python
# In modify_attn_features():
# Apply mask to query blending
mask = load_mask(...)  # 0 = no style, 1 = full style
gamma_masked = gamma * mask + 1.0 * (1 - mask)
query_modified = gamma_masked * query_content + (1 - gamma_masked) * query_stylized
```

---

### Technique 3: Style Interpolation

Blend multiple style references:

```python
# Collect features from multiple styles
style_features_A = invert_style(style_A)
style_features_B = invert_style(style_B)

# Interpolate K, V features
alpha = 0.5  # blend ratio
K_blended = alpha * style_features_A['K'] + (1-alpha) * style_features_B['K']
V_blended = alpha * style_features_A['V'] + (1-alpha) * style_features_B['V']

# Use blended features for injection
```

---

### Technique 4: Time-Varying Parameters

Change γ or τ across timesteps for progressive effects:

```python
def get_dynamic_gamma(timestep, total_steps):
    """Start with strong style, progressively preserve content"""
    progress = timestep / total_steps
    return 0.3 + 0.6 * progress  # γ: 0.3 → 0.9

# In denoise loop:
for t in timesteps:
    gamma_t = get_dynamic_gamma(t, len(timesteps))
    # Use gamma_t for current step
```

---

## Quality Assessment Checklist

After generating a result, evaluate:

### Content Fidelity
- [ ] Main subject is recognizable
- [ ] Spatial layout matches original
- [ ] Important details preserved
- [ ] Proportions are correct

### Style Transfer Quality
- [ ] Texture patterns from style visible
- [ ] Color palette resembles style
- [ ] Artistic characteristics captured
- [ ] Style elements well-integrated

### Technical Quality
- [ ] No obvious artifacts
- [ ] Sharp and detailed (not blurry)
- [ ] Natural transitions
- [ ] Consistent style application

### Overall Assessment
- [ ] Achieves desired balance
- [ ] Visually appealing
- [ ] Meets use case requirements

---

## Batch Processing Script

Process multiple images with same settings:

```bash
#!/bin/bash
# batch_styleid.sh

STYLE="artworks/picasso.jpg"
CONTENT_DIR="photos/"
OUTPUT_DIR="results/"

for content in $CONTENT_DIR/*.jpg; do
    filename=$(basename "$content" .jpg)
    python run_styleid_diffusers.py \
        --cnt_fn "$content" \
        --sty_fn "$STYLE" \
        --gamma 0.75 \
        --T 1.5 \
        --save_dir "$OUTPUT_DIR/$filename/"
done
```

Usage:
```bash
chmod +x batch_styleid.sh
./batch_styleid.sh
```

---

## Performance Optimization Tips

### 1. Reduce DDIM Steps
```bash
# Default: 50 steps
# Try: 20-30 steps for faster inference
--ddim_steps 20
```

### 2. Use Precomputed Features
```bash
# Save inversion features for reuse
# (Automatic in original implementation)
# Saves time when experimenting with same images
```

### 3. Lower Resolution for Experiments
```python
# In code, resize images before encoding
image = image.resize((256, 256))  # Instead of 512×512
```

### 4. Use Mixed Precision
```python
# Add to code
import torch
torch.set_default_dtype(torch.float16)
```

---

## Style Image Selection Guidelines

### Good Style Images:
✓ Clear, distinctive textures
✓ Consistent artistic style
✓ High quality, sharp details
✓ Strong color palette
✓ Recognizable patterns

### Poor Style Images:
✗ Blurry or low resolution
✗ Mixed/inconsistent styles
✗ Photorealistic (not artistic)
✗ Too simple/minimal
✗ Heavy noise or compression

### Recommended Sources:
- WikiArt (classical paintings)
- Modern digital art platforms
- Museum collections
- Artist portfolios

---

## Content Image Preparation

### Best Practices:
1. **Resolution:** 512×512 or 1024×1024
2. **Format:** PNG or high-quality JPG
3. **Composition:** Clear subject, good contrast
4. **Quality:** Sharp, well-lit, not compressed

### Pre-processing Tips:
```python
# Enhance before style transfer
from PIL import Image, ImageEnhance

image = Image.open('photo.jpg')

# Increase sharpness
enhancer = ImageEnhance.Sharpness(image)
image = enhancer.enhance(1.5)

# Adjust contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.2)

image.save('enhanced.jpg')
```

---

## Experimental Ideas

### Experiment 1: Layer Ablation Study
Test which layers contribute most to style transfer:

```bash
# Test each layer individually
for layer in 7 8 9 10 11; do
    python run_styleid_diffusers.py ... --layers $layer
done

# Test combinations
python run_styleid_diffusers.py ... --layers 7 8 9
python run_styleid_diffusers.py ... --layers 9 10 11
```

### Experiment 2: Parameter Sweep
Create a grid of results with different γ values:

```bash
for gamma in 0.1 0.3 0.5 0.7 0.9; do
    python run_styleid_diffusers.py ... --gamma $gamma
done
```

### Experiment 3: Style Mixing
Combine features from different styles at different layers:

```python
# Layer 7-8: Style A
# Layer 9-11: Style B
# Creates unique hybrid effects
```

### Experiment 4: Temporal Coherence (Video)
Apply to video frames with feature caching:

```python
# Cache style features once
# Process each frame with same features
# Maintains consistency across frames
```

---

## Troubleshooting Decision Tree

```
Is the output satisfactory?
│
├─ NO: What's the main issue?
│   │
│   ├─ Content lost
│   │   └─ Increase gamma (0.85-0.95)
│   │       └─ Reduce injection layers
│   │
│   ├─ Style too weak
│   │   └─ Decrease gamma (0.2-0.4)
│   │       └─ Increase injection layers
│   │
│   ├─ Blurry/soft
│   │   └─ Increase temperature (1.8-2.0)
│   │
│   ├─ Colors wrong
│   │   └─ Toggle --without_init_adain
│   │       └─ Or adjust style image colors
│   │
│   └─ Artifacts/distortions
│       └─ Reduce temperature (1.0-1.3)
│           └─ Return to balanced settings
│
└─ YES: Great! Save settings for similar tasks
```

---

## Final Tips

1. **Start Conservative:** Begin with default settings, adjust gradually
2. **Keep Notes:** Document successful parameter combinations
3. **Test Multiple Styles:** Not all styles work equally well with all content
4. **Be Patient:** Quality results may require several iterations
5. **Understand Trade-offs:** Every parameter affects multiple aspects
6. **Visual Comparison:** Always compare with content and style side-by-side
7. **Subjective Judgment:** Style transfer is artistic—trust your eye!

---

## Resources

### Original Paper:
- "Style Injection in Diffusion: A Training-free Approach" (CVPR 2024)
- ArXiv: https://arxiv.org/abs/2312.09008

### Code Repository:
- GitHub: https://github.com/jiwoogit/StyleID

### Related Work:
- Prompt-to-Prompt (cross-attention manipulation)
- InST (inversion-based style transfer)
- DiffStyle (early diffusion-based style transfer)

---

**Happy Style Transferring! 🎨✨**
