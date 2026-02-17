"""
Patches for the DDIM sampler to incorporate ControlNet during the denoising (reverse) pass.

Place this file in the root of the StyleID repo.
"""

import torch
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


def patch_ddim_for_controlnet(sampler, controlnet_wrapper, controlnet_cond_tensor,
                                conditioning_scale=1.0, text_embeddings=None):
    """
    Patch a DDIMSampler instance so that its sample() method uses ControlNet
    during the reverse (denoising) pass.
    
    Args:
        sampler: DDIMSampler instance
        controlnet_wrapper: ControlNetWrapper instance
        controlnet_cond_tensor: [1, 3, 512, 512] condition tensor (e.g., depth map)
        conditioning_scale: float, strength of ControlNet
        text_embeddings: text embeddings for ControlNet (uses uncond if None)
    """
    import types
    
    # Store references on the sampler for access in patched methods
    sampler._cn_wrapper = controlnet_wrapper
    sampler._cn_cond = controlnet_cond_tensor
    sampler._cn_scale = conditioning_scale
    sampler._cn_text_emb = text_embeddings
    
    original_p_sample = sampler.p_sample_ddim
    
    @torch.no_grad()
    def p_sample_ddim_with_cn(self, x, c, t, index, negative_conditioning=None,
                              repeat_noise=False, use_original_steps=False, 
                              quantize_denoised=False,
                              temperature=1., noise_dropout=0., score_corrector=None,
                              corrector_kwargs=None,
                              unconditional_guidance_scale=1., 
                              unconditional_conditioning=None,
                              injected_features=None, negative_prompt_alpha=1.0,
                              style_guidance_scale=1., style_loss=None, style_img=None,
                              content_guidance_scale=1.):
        
        b, *_, device = *x.shape, x.device
        
        # ---- Compute ControlNet residuals for this timestep ----
        cn_residuals = None
        if self._cn_wrapper is not None:
            # We need encoder_hidden_states for ControlNet
            # Use unconditional embedding (empty text), same as StyleID uses
            if self._cn_text_emb is not None:
                cn_text = self._cn_text_emb
            elif unconditional_conditioning is not None:
                # unconditional_conditioning is the text embedding
                cn_text = unconditional_conditioning
                if isinstance(cn_text, dict):
                    cn_text = cn_text.get('c_crossattn', [cn_text])[0]
                if isinstance(cn_text, list):
                    cn_text = torch.cat(cn_text, dim=1)
            else:
                cn_text = self.model.get_learned_conditioning([""])
                
            # Ensure shapes match
            cn_cond = self._cn_cond
            if cn_cond.shape[0] != x.shape[0]:
                cn_cond = cn_cond.repeat(x.shape[0], 1, 1, 1)
            if cn_text.shape[0] != x.shape[0]:
                cn_text = cn_text.repeat(x.shape[0], 1, 1)
            
            down_samples, mid_sample = self._cn_wrapper.get_residuals(
                sample=x.to(self._cn_wrapper.dtype),
                timestep=t[0] if t.dim() > 0 else t,  # scalar timestep
                encoder_hidden_states=cn_text.to(self._cn_wrapper.dtype),
                controlnet_cond=cn_cond,
                conditioning_scale=self._cn_scale,
            )
            # Cast back to model dtype
            down_samples = [d.to(x.dtype) for d in down_samples]
            mid_sample = mid_sample.to(x.dtype)
            cn_residuals = (down_samples, mid_sample)
        
        # ---- Original p_sample_ddim logic, but with controlnet_residuals ----
        if negative_conditioning is not None:
            # Not used in StyleID's default path, skip for brevity
            return original_p_sample(
                x, c, t, index, negative_conditioning=negative_conditioning,
                repeat_noise=repeat_noise, use_original_steps=use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                injected_features=injected_features,
                negative_prompt_alpha=negative_prompt_alpha,
                style_guidance_scale=style_guidance_scale,
                style_loss=style_loss, style_img=style_img,
                content_guidance_scale=content_guidance_scale,
            )
        
        # StyleID default: no text conditioning, uses unconditional only
        if c is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            
            # Double the residuals for CFG (same residuals for both uncond and cond)
            cn_res_doubled = None
            if cn_residuals is not None:
                down_doubled = [torch.cat([d, d]) for d in cn_residuals[0]]
                mid_doubled = torch.cat([cn_residuals[1], cn_residuals[1]])
                cn_res_doubled = (down_doubled, mid_doubled)
            
            e_t_uncond, e_t = self.model.apply_model(
                x_in, t_in, c_in,
                injected_features=injected_features,
                controlnet_residuals=cn_res_doubled
            ).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) \
                if unconditional_guidance_scale != 1 else e_t
        else:
            # This is the path StyleID actually takes (guidance_scale=0, c=None)
            x_in = x
            t_in = t
            c_in = unconditional_conditioning
            e_t_uncond = self.model.apply_model(
                x_in, t_in, c_in,
                injected_features=injected_features,
                controlnet_residuals=cn_residuals
            )
            e_t = e_t_uncond

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sqrt_alphas_cumprod = self.model.sqrt_alphas_cumprod if use_original_steps else self.ddim_sqrt_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    sampler.p_sample_ddim = types.MethodType(p_sample_ddim_with_cn, sampler)
    print(f"[ControlNet] DDIM sampler patched (scale={conditioning_scale}).")


def unpatch_ddim(sampler):
    """Remove ControlNet patches from sampler (for inversion passes)."""
    if hasattr(sampler, '_cn_wrapper'):
        sampler._cn_wrapper = None
        sampler._cn_cond = None
        sampler._cn_scale = 1.0
        sampler._cn_text_emb = None
