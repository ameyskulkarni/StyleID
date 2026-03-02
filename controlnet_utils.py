"""
ControlNet integration for StyleID.

This module provides:
1. ControlNet model loading (from diffusers-format checkpoints)
2. Depth map extraction from content images (using MiDaS)  
3. ControlNet forward pass that produces residuals for the UNet

Place this file in the root of the StyleID repo (next to run_styleid.py).

Supported condition types (phase 1: depth only, extensible):
  - depth: MiDaS depth estimation
  - [future] canny, normal, seg, hed, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from enum import Enum


class ConditionType(str, Enum):
    DEPTH = "depth"
    CANNY = "canny"
    NORMAL = "normal"
    SEG = "seg"
    HED = "hed"


# ─────────────────────────────────────────────────
# 1. Condition extraction (depth map, etc.)
# ─────────────────────────────────────────────────

def extract_condition(image_np, condition_type="depth", device="cuda", **kwargs):
    """
    Extract a condition map from a content image.

    Args:
        image_np: numpy array, HxWx3, uint8 (RGB)
        condition_type: one of ConditionType values
        device: torch device
    Returns:
        condition_image: PIL Image (512x512, RGB) ready for ControlNet
    """
    if condition_type == ConditionType.DEPTH:
        return _extract_depth(image_np, device)
    elif condition_type == ConditionType.CANNY:
        return _extract_canny(image_np, **kwargs)
    elif condition_type == ConditionType.NORMAL:
        return _extract_normal(image_np, device)
    elif condition_type == ConditionType.HED:
        return _extract_hed(image_np, device)
    elif condition_type == ConditionType.SEG:
        return _extract_seg(image_np, device)
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")


def _extract_depth(image_np, device="cuda"):
    """Extract depth map using MiDaS via transformers pipeline."""
    from transformers import pipeline as hf_pipeline
    
    depth_estimator = hf_pipeline('depth-estimation', device=device)
    pil_img = Image.fromarray(image_np).resize((512, 512))
    result = depth_estimator(pil_img)
    depth = result['depth']  # PIL Image, single channel
    depth = np.array(depth)
    
    # Normalize to 0-255
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    depth = depth.astype(np.uint8)
    
    # Convert to 3-channel (ControlNet expects RGB)
    depth_3ch = np.stack([depth, depth, depth], axis=2)
    
    del depth_estimator
    torch.cuda.empty_cache()
    
    return Image.fromarray(depth_3ch)


def _extract_canny(image_np, low_threshold=100, high_threshold=200):
    """Extract Canny edges (no GPU needed)."""
    import cv2
    img = cv2.resize(image_np, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_3ch)


def _extract_normal(image_np, device="cuda"):
    """Extract normal map. Placeholder for future use."""
    raise NotImplementedError("Normal map extraction not yet implemented. Coming in a future phase.")


def _extract_hed(image_np, device="cuda"):
    """Extract HED soft edges. Placeholder for future use."""
    raise NotImplementedError("HED extraction not yet implemented. Coming in a future phase.")


def _extract_seg(image_np, device="cuda"):
    """Extract segmentation map. Placeholder for future use."""
    raise NotImplementedError("Segmentation extraction not yet implemented. Coming in a future phase.")


# ─────────────────────────────────────────────────
# 2. ControlNet model wrapper (for original LDM codebase)
# ─────────────────────────────────────────────────

class ControlNetWrapper:
    """
    Loads a ControlNet from HuggingFace diffusers format and runs it
    to produce residuals that are added to the SD UNet's skip connections.
    
    ControlNet architecture: 
      - Copies the encoder half of the UNet (input_blocks + middle_block)
      - Adds zero_convs after each block
      - Produces residuals for each skip connection + middle block
      
    Integration with StyleID's LDM UNet:
      - The residuals are added to `hs` (skip connections) before the decoder uses them
      - The middle block residual is added to `h` after middle_block
    """
    
    def __init__(self, controlnet_model_id, device="cuda", dtype=torch.float16):
        """
        Args:
            controlnet_model_id: HuggingFace model ID, e.g. "lllyasviel/sd-controlnet-depth"
            device: torch device
            dtype: precision
        """
        from diffusers import ControlNetModel
        
        print(f"Loading ControlNet from {controlnet_model_id}...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id, torch_dtype=dtype
        )
        self.controlnet.to(device)
        self.controlnet.eval()
        self.device = device
        self.dtype = dtype
        print("ControlNet loaded.")
    
    def prepare_condition(self, condition_image, device=None, dtype=None):
        """
        Convert a PIL condition image to the tensor format ControlNet expects.
        
        Args:
            condition_image: PIL Image (512x512, RGB)
        Returns:
            condition_tensor: [1, 3, 512, 512] normalized to [0, 1]
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
            
        img = condition_image.resize((512, 512))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(device=device, dtype=dtype)
    
    def get_residuals(self, sample, timestep, encoder_hidden_states, controlnet_cond,
                      conditioning_scale=1.0):
        """
        Run ControlNet forward pass to get residuals.
        
        Args:
            sample: noisy latent [B, 4, 64, 64]
            timestep: current timestep (scalar or tensor)
            encoder_hidden_states: text conditioning [B, 77, 768]
            controlnet_cond: condition image tensor [B, 3, 512, 512]
            conditioning_scale: strength of ControlNet influence
            
        Returns:
            down_block_res_samples: list of tensors for skip connections
            mid_block_res_sample: tensor for middle block
        """
        with torch.no_grad():
            result = self.controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                return_dict=True,
            )
        
        down_block_res_samples = result.down_block_res_samples
        mid_block_res_sample = result.mid_block_res_sample
        
        return down_block_res_samples, mid_block_res_sample
    
    def to_cpu(self):
        """Move ControlNet to CPU to free VRAM."""
        self.controlnet.to("cpu")
        torch.cuda.empty_cache()
    
    def to_device(self):
        """Move ControlNet back to GPU."""
        self.controlnet.to(self.device)


# ─────────────────────────────────────────────────
# 3. Model ID mapping for different condition types
# ─────────────────────────────────────────────────

CONTROLNET_MODEL_IDS = {
    ConditionType.DEPTH: "lllyasviel/sd-controlnet-depth",
    ConditionType.CANNY: "lllyasviel/sd-controlnet-canny",
    ConditionType.NORMAL: "lllyasviel/sd-controlnet-normal",
    ConditionType.SEG: "lllyasviel/sd-controlnet-seg",
    ConditionType.HED: "lllyasviel/sd-controlnet-hed",
}

CONTROLNET_V11_MODEL_IDS = {
    ConditionType.DEPTH: "lllyasviel/control_v11f1p_sd15_depth",
    ConditionType.CANNY: "lllyasviel/control_v11p_sd15_canny",
    ConditionType.NORMAL: "lllyasviel/control_v11p_sd15_normalbae",
    ConditionType.SEG: "lllyasviel/control_v11p_sd15_seg",
    ConditionType.HED: "lllyasviel/control_v11p_sd15_softedge",
}


def get_controlnet_model_id(condition_type, version="1.0"):
    """Get the HuggingFace model ID for a given condition type."""
    if version == "1.1":
        return CONTROLNET_V11_MODEL_IDS[condition_type]
    return CONTROLNET_MODEL_IDS[condition_type]
