"""Shared traceable wrappers for VibeVoice TTS/ASR components.

Each wrapper isolates a component for torch.jit.trace, removing
dynamic control flow and returning only tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TraceableVAEDecoder(nn.Module):
    """Acoustic VAE decoder: latent (B, vae_dim, T) -> audio (B, 1, samples)."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class TraceableDiffusionHead(nn.Module):
    """Single denoising step: (noisy, timestep, condition) -> predicted noise."""

    def __init__(self, prediction_head: nn.Module) -> None:
        super().__init__()
        self.prediction_head = prediction_head

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.prediction_head(noisy_latent, timestep, condition=condition)


class TraceableAcousticConnector(nn.Module):
    """Projects acoustic latent to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.connector(latent)


class TraceableSemanticConnector(nn.Module):
    """Projects semantic features to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.connector(features)


class TraceableLMHead(nn.Module):
    """LM head: hidden_state -> next-token logits."""

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_state)


class TraceableDiffusionLoopCFG(nn.Module):
    """Full DPM-Solver++ 2M diffusion loop with CFG, fused into one model.

    Replaces N*2 separate diffusion head calls with a single forward pass.
    Schedule constants are baked in at construction time; CFG scale is a
    runtime input for flexibility.
    Input: (noise, condition, neg_condition, cfg_scale) -> latent.
    """

    def __init__(
        self,
        prediction_head: nn.Module,
        num_steps: int = 10,
        cfg_scale: float = 1.3,  # unused, kept for API compat
    ) -> None:
        super().__init__()
        self.prediction_head = prediction_head
        self.num_steps = num_steps

        # Cosine noise schedule (same as diffusion.py)
        DDPM_STEPS = 1000
        steps = np.arange(DDPM_STEPS + 1, dtype=np.float64) / DDPM_STEPS
        ac = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = (ac / ac[0])[:DDPM_STEPS]
        alpha = np.sqrt(alphas_cumprod)
        sigma = np.sqrt(1.0 - alphas_cumprod)
        lam = np.log(alpha / np.maximum(sigma, 1e-10))

        # DPM-Solver++ 2M timestep schedule
        t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)

        # Pre-compute per-step constants (stored as Python floats for tracing)
        self._steps = []
        for i in range(num_steps):
            s = int(t_schedule[i])
            t = int(t_schedule[i + 1])
            s_prev = int(t_schedule[i - 1]) if i > 0 else s

            lam_s = float(lam[s])
            lam_t = float(lam[max(t, 0)])
            h = lam_t - lam_s

            is_last = (i == num_steps - 1)
            is_second_last = (i == num_steps - 2)
            lower_order_final = is_last and num_steps < 15
            lower_order_second = is_second_last and num_steps < 15
            use_first_order = (i < 1) or lower_order_final or lower_order_second

            lam_s_prev = float(lam[s_prev])
            h_prev = lam_s - lam_s_prev
            r = h_prev / h if abs(h) > 1e-10 else 0.0

            self._steps.append({
                "timestep": float(s),
                "alpha_s": float(alpha[s]),
                "sigma_s": float(sigma[s]),
                "sigma_t": float(sigma[max(t, 0)]),
                "alpha_t": float(alpha[max(t, 0)]),
                "sigma_ratio": float(sigma[max(t, 0)] / sigma[s]),
                "alpha_t_expm1_neg_h": float(alpha[max(t, 0)]) * float(np.expm1(-h)),
                "first_order": use_first_order,
                "r": r,
            })

    def forward(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        neg_condition: torch.Tensor,
        cfg_scale: torch.Tensor,
    ) -> torch.Tensor:
        sample = noise
        x0_list: list[torch.Tensor] = []

        for i in range(self.num_steps):
            step = self._steps[i]
            timestep = sample.new_tensor([step["timestep"]])

            # CFG: two forward passes through diffusion head
            v_cond = self.prediction_head(sample, timestep, condition=condition)
            v_uncond = self.prediction_head(sample, timestep, condition=neg_condition)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            # x0 prediction
            x0 = step["alpha_s"] * sample - step["sigma_s"] * v
            x0_list.append(x0)

            # DPM-Solver++ 2M update
            if step["first_order"]:
                D = x0_list[-1]
            else:
                D = x0_list[-1] + 0.5 * (1.0 / step["r"]) * (x0_list[-1] - x0_list[-2])

            sample = step["sigma_ratio"] * sample - step["alpha_t_expm1_neg_h"] * D

        return sample


class TraceableStreamingVAEDecoder(nn.Module):
    """Stateful streaming VAE decoder with registered buffer conv caches.

    Each SConv1d/SConvTranspose1d layer's causal context buffer is a
    registered buffer, making it compatible with CoreML ct.StateType export.
    Processes one frame (T=1) per call, carrying state across frames.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

        from vibevoice.modular.modular_vibevoice_tokenizer import SConv1d, SConvTranspose1d

        self.cache_layers = []  # [(name, module, in_channels, context_size, is_transpose)]
        depths = decoder.depths
        ratios = decoder.ratios

        # Execution order mirrors forward_features + head
        stem = decoder.upsample_layers[0][0]
        self.cache_layers.append(("stem", stem, stem.in_channels, stem.context_size, False))

        for b in range(depths[0]):
            mod = decoder.stages[0][b].mixer.conv
            self.cache_layers.append((f"s0_b{b}", mod, mod.in_channels, mod.context_size, False))

        for s in range(len(ratios)):
            up = decoder.upsample_layers[s + 1][0]
            self.cache_layers.append(
                (f"up{s}", up, up.in_channels, up.context_size, True))
            for b in range(depths[s + 1]):
                mod = decoder.stages[s + 1][b].mixer.conv
                self.cache_layers.append(
                    (f"s{s+1}_b{b}", mod, mod.in_channels, mod.context_size, False))

        head = decoder.head
        self.cache_layers.append(("head", head, head.in_channels, head.context_size, False))

        # Register state buffers (fp16, like LM KV caches)
        for name, _, ch, ctx, _ in self.cache_layers:
            self.register_buffer(
                f"cache_{name}",
                torch.zeros(1, ch, ctx, dtype=torch.float16),
            )

    def _sconv1d_streaming(self, mod, x, cache_name):
        """Streaming SConv1d: read cache, prepend, conv, write cache."""
        cache_buf = getattr(self, cache_name).float()
        inp = torch.cat([cache_buf, x], dim=2)
        out = mod.conv(inp)
        ctx = mod.context_size
        getattr(self, cache_name)[:] = inp[:, :, -ctx:].half()
        return out

    def _sconvtr1d_streaming(self, mod, x, cache_name):
        """Streaming SConvTranspose1d: read cache, transposed conv, trim, write cache."""
        import math
        cache_buf = getattr(self, cache_name).float()
        full_input = torch.cat([cache_buf, x], dim=2)
        full_output = mod.convtr(full_input)

        padding_right = math.ceil(mod.padding_total * mod.trim_right_ratio)
        padding_left = mod.padding_total - padding_right
        if padding_left + padding_right > 0:
            end = full_output.shape[2] - padding_right if padding_right > 0 else full_output.shape[2]
            full_output = full_output[:, :, padding_left:end]

        new_samples = x.shape[2] * mod.stride
        out = full_output[:, :, -new_samples:]

        ctx = mod.context_size
        getattr(self, cache_name)[:] = full_input[:, :, -ctx:].half()
        return out

    def _block_forward(self, block, x, conv_mod, cache_name):
        """Block1D forward with stateful conv cache."""
        residual = x
        x_normed = block.norm(x)
        x_conv = self._sconv1d_streaming(conv_mod, x_normed, cache_name)
        if block.gamma is not None:
            x_conv = x_conv * block.gamma.unsqueeze(-1)
        x = residual + x_conv

        residual = x
        xt = block.ffn_norm(x)
        xt = xt.permute(0, 2, 1)
        xt = block.ffn(xt)
        xt = xt.permute(0, 2, 1)
        if block.ffn_gamma is not None:
            xt = xt * block.ffn_gamma.unsqueeze(-1)
        x = residual + xt
        return x

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward one frame, reading/writing internal cache buffers.

        Args:
            latent: (1, vae_dim, T)

        Returns:
            audio: (1, 1, samples)
        """
        x = latent
        cache_idx = 0
        depths = self.decoder.depths

        # Stage 0: stem
        x = self._sconv1d_streaming(self.cache_layers[0][1], x, f"cache_{self.cache_layers[0][0]}")
        cache_idx = 1

        # Stage 0 blocks
        for b in range(depths[0]):
            name, mod, _, _, _ = self.cache_layers[cache_idx]
            block = self.decoder.stages[0][b]
            x = self._block_forward(block, x, mod, f"cache_{name}")
            cache_idx += 1

        # Stages 1..N: upsample + blocks
        for s in range(len(self.decoder.ratios)):
            name, up_mod, _, _, _ = self.cache_layers[cache_idx]
            x = self._sconvtr1d_streaming(up_mod, x, f"cache_{name}")
            cache_idx += 1

            for b in range(depths[s + 1]):
                name, mod, _, _, _ = self.cache_layers[cache_idx]
                block = self.decoder.stages[s + 1][b]
                x = self._block_forward(block, x, mod, f"cache_{name}")
                cache_idx += 1

        # Final norm + head
        x = self.decoder.norm(x)
        name, head_mod, _, _, _ = self.cache_layers[cache_idx]
        x = self._sconv1d_streaming(head_mod, x, f"cache_{name}")

        return x
