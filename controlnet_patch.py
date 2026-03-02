"""
Monkey-patch for the StyleID UNet to support ControlNet residuals.

This patches UNetModel.forward() to accept and apply ControlNet residuals
(added to skip connections and middle block output) without modifying
the original openaimodel.py.

Place this file in the root of the StyleID repo (next to run_styleid.py).
"""

import torch
import torch as th
from ldm.modules.diffusionmodules.openaimodel import UNetModel, timestep_embedding


def make_controlnet_forward(original_forward):
    """
    Create a new forward method that wraps the original UNet forward
    to accept ControlNet residuals.
    """
    def forward_with_controlnet(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        injected_features=None,
        controlnet_residuals=None,  # NEW: (down_block_res_samples, mid_block_res_sample) or None
        **kwargs
    ):
        if controlnet_residuals is None:
            # No ControlNet — use original forward
            return original_forward(
                x, timesteps=timesteps, context=context, y=y,
                injected_features=injected_features, **kwargs
            )
        
        down_block_res_samples, mid_block_res_sample = controlnet_residuals
        
        # ---- Replicate the UNet forward with ControlNet residuals injected ----
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        
        # Encoder (input_blocks) — add ControlNet residuals to skip connections
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            # Add ControlNet residual to this skip connection
            if i < len(down_block_res_samples):
                h = h + down_block_res_samples[i]
            hs.append(h)

        # Middle block
        h = self.middle_block(h, emb, context)
        # Add ControlNet middle block residual
        if mid_block_res_sample is not None:
            h = h + mid_block_res_sample

        # Decoder (output_blocks) — same as original, with injected_features
        module_i = 0
        for module in self.output_blocks:
            self_attn_q_injected = None
            self_attn_k_injected = None
            self_attn_v_injected = None
            out_layers_injected = None
            injection_config = None
            q_feature_key = f'output_block_{module_i}_self_attn_q'
            k_feature_key = f'output_block_{module_i}_self_attn_k'
            v_feature_key = f'output_block_{module_i}_self_attn_v'
            out_layers_feature_key = f'output_block_{module_i}_out_layers'
            t_scale_key = f'output_block_{module_i}_self_attn_s'
            config_key = f'config'

            if injected_features is not None and q_feature_key in injected_features:
                self_attn_q_injected = injected_features[q_feature_key]
            if injected_features is not None and k_feature_key in injected_features:
                self_attn_k_injected = injected_features[k_feature_key]
            if injected_features is not None and v_feature_key in injected_features:
                self_attn_v_injected = injected_features[v_feature_key]
            if injected_features is not None and out_layers_feature_key in injected_features:
                out_layers_injected = injected_features[out_layers_feature_key]
            if injected_features is not None:
                injection_config = injected_features[config_key]
                if t_scale_key in injected_features:
                    injection_config['T'] = injected_features[t_scale_key]

            h = th.cat([h, hs.pop()], dim=1)
            h = module(h,
                       emb,
                       context,
                       self_attn_q_injected=self_attn_q_injected,
                       self_attn_k_injected=self_attn_k_injected,
                       self_attn_v_injected=self_attn_v_injected,
                       out_layers_injected=out_layers_injected,
                       injection_config=injection_config,)
            module_i += 1

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
    
    return forward_with_controlnet


def patch_unet_for_controlnet(unet_model):
    """
    Monkey-patch a UNetModel instance so its forward() accepts
    controlnet_residuals=(down_samples, mid_sample).
    
    Call this once after loading the model.
    """
    import types
    original_forward = unet_model.forward
    new_forward = make_controlnet_forward(original_forward)
    unet_model.forward = types.MethodType(new_forward, unet_model)
    print("[ControlNet] UNet patched to accept ControlNet residuals.")


def patch_diffusion_wrapper(model):
    """
    Patch the DiffusionWrapper (model.model) to pass controlnet_residuals
    through to the diffusion_model (UNet).
    
    The DiffusionWrapper.forward() needs to accept and forward controlnet_residuals.
    """
    import types
    
    original_wrapper_forward = model.model.forward
    
    def wrapper_forward_with_cn(self, x, t, c_concat=None, c_crossattn=None,
                                 injected_features=None, controlnet_residuals=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, controlnet_residuals=controlnet_residuals)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t,
                                        injected_features=injected_features,
                                        controlnet_residuals=controlnet_residuals)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc,
                                        injected_features=injected_features,
                                        controlnet_residuals=controlnet_residuals)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc,
                                        injected_features=injected_features,
                                        controlnet_residuals=controlnet_residuals)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc,
                                        injected_features=injected_features,
                                        controlnet_residuals=controlnet_residuals)
        else:
            raise NotImplementedError()
        return out
    
    model.model.forward = types.MethodType(wrapper_forward_with_cn, model.model)
    print("[ControlNet] DiffusionWrapper patched to forward ControlNet residuals.")


def patch_apply_model(model):
    """
    Patch LatentDiffusion.apply_model() to accept and forward controlnet_residuals.
    """
    import types
    
    original_apply = model.apply_model
    
    def apply_model_with_cn(self, x_noisy, t, cond, injected_features=None,
                            controlnet_residuals=None):
        # The original apply_model unpacks cond into c_concat / c_crossattn
        # and calls self.model (DiffusionWrapper)
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]

        key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
        
        if isinstance(cond, dict):
            c_crossattn = cond.get('c_crossattn', None)
            c_concat = cond.get('c_concat', None)
        else:
            c_crossattn = cond if key == 'c_crossattn' else None
            c_concat = cond if key == 'c_concat' else None
            
        return self.model(x_noisy, t,
                         c_concat=c_concat,
                         c_crossattn=c_crossattn,
                         injected_features=injected_features,
                         controlnet_residuals=controlnet_residuals)
    
    model.apply_model = types.MethodType(apply_model_with_cn, model)
    print("[ControlNet] apply_model patched to forward ControlNet residuals.")
