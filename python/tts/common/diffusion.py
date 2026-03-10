"""Shared diffusion sampling for VibeVoice TTS models.

Supports DDPM and DPM-Solver++ 2M with v-prediction and cosine schedule.
All models share vae_dim=64 and the same noise schedule.
"""

from __future__ import annotations

import numpy as np

DDPM_STEPS = 1000
VAE_DIM = 64


def cosine_alphas_cumprod(num_steps: int = DDPM_STEPS) -> np.ndarray:
    steps = np.arange(num_steps + 1, dtype=np.float64) / num_steps
    ac = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    return (ac / ac[0])[:num_steps].astype(np.float32)


ALPHAS_CUMPROD = cosine_alphas_cumprod(DDPM_STEPS)

# Precompute schedule values (float64 for precision)
_AC64 = ALPHAS_CUMPROD.astype(np.float64)
_ALPHA = np.sqrt(_AC64)
_SIGMA = np.sqrt(1.0 - _AC64)
_LAMBDA = np.log(_ALPHA / np.maximum(_SIGMA, 1e-10))


def make_timesteps(num_steps: int) -> np.ndarray:
    """Create evenly-spaced timestep schedule from T-1 to 0."""
    return np.linspace(0, DDPM_STEPS - 1, num_steps).astype(np.int64)[::-1].copy()


def ddpm_step_v(sample, v_pred, alpha_t, alpha_prev):
    """Single DDPM step with v-prediction."""
    sqrt_alpha = np.sqrt(alpha_t)
    sqrt_one_minus_alpha = np.sqrt(max(1 - alpha_t, 1e-8))
    pred_x0 = sqrt_alpha * sample - sqrt_one_minus_alpha * v_pred
    pred_eps = sqrt_one_minus_alpha * sample + sqrt_alpha * v_pred
    return np.sqrt(alpha_prev) * pred_x0 + np.sqrt(max(1 - alpha_prev, 1e-8)) * pred_eps


def ddpm_sample(diffusion_fn, condition, num_steps: int = 20, seed=None):
    """DDPM denoising (v-prediction, deterministic / eta=0)."""
    timesteps = make_timesteps(num_steps)
    rng = np.random.RandomState(seed)
    sample = rng.randn(1, VAE_DIM).astype(np.float32)
    for i, t in enumerate(timesteps):
        v_pred = diffusion_fn(sample, np.array([float(t)], dtype=np.float32), condition)
        alpha_t = float(ALPHAS_CUMPROD[int(t)])
        alpha_prev = float(ALPHAS_CUMPROD[int(timesteps[i + 1])]) if i < len(timesteps) - 1 else 1.0
        sample = ddpm_step_v(sample, v_pred, alpha_t, alpha_prev)
    return sample


def dpm_solver_2m_sample(diffusion_fn, condition, num_steps: int = 10, seed=None):
    """DPM-Solver++ 2M for v-prediction (2nd order multistep ODE solver).

    Converges faster than DDPM: 10 DPM-Solver steps ≈ 20 DDPM steps quality.
    Uses lower_order_final: when num_steps < 15, last 1-2 steps use first-order
    updates for numerical stability.
    """
    rng = np.random.RandomState(seed)
    t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)

    sample = rng.randn(1, VAE_DIM).astype(np.float32)
    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        v = diffusion_fn(sample, np.array([float(s)], dtype=np.float32), condition)
        x0 = float(_ALPHA[s]) * sample - float(_SIGMA[s]) * v
        x0_list.append(x0)

        lam_s = float(_LAMBDA[s])
        lam_t = float(_LAMBDA[max(t, 0)])
        h = lam_t - lam_s

        is_last = (i == num_steps - 1)
        is_second_last = (i == num_steps - 2)
        lower_order_final = is_last and num_steps < 15
        lower_order_second = is_second_last and num_steps < 15
        use_first_order = len(x0_list) < 2 or lower_order_final or lower_order_second

        if use_first_order:
            D = x0_list[-1]
        else:
            s_prev = int(t_schedule[i - 1])
            lam_s_prev = float(_LAMBDA[s_prev])
            h_prev = lam_s - lam_s_prev
            r = h_prev / h
            D0 = x0_list[-1]
            D1 = (1.0 / r) * (x0_list[-1] - x0_list[-2])
            D = D0 + 0.5 * D1

        sample = (float(_SIGMA[t]) / float(_SIGMA[s])) * sample \
               - float(_ALPHA[t]) * float(np.expm1(-h)) * D

    return sample


def make_batched_cfg_fn(diffusion_fn_b2, neg_condition, cfg_scale: float):
    """Wrap a B=2 diffusion function with batched Classifier-Free Guidance.

    Instead of two separate B=1 calls, concatenates positive and negative
    inputs along B=0, calls once with B=2, splits, and applies CFG.
    Formula: v = v_uncond + scale * (v_cond - v_uncond)
    """
    if cfg_scale <= 1.0:
        # With no guidance, just call with B=1 shape (extract from B=2 fn not possible)
        raise ValueError("make_batched_cfg_fn requires cfg_scale > 1.0")

    def guided_fn(sample, timestep, condition):
        # sample: (1, 64), timestep: (1,), condition: (1, H)
        sample_b2 = np.concatenate([sample, sample], axis=0)       # (2, 64)
        timestep_b2 = np.concatenate([timestep, timestep], axis=0)  # (2,)
        cond_b2 = np.concatenate([condition, neg_condition], axis=0)  # (2, H)
        v_b2 = diffusion_fn_b2(sample_b2, timestep_b2, cond_b2)    # (2, 64)
        v_cond = v_b2[0:1]   # (1, 64)
        v_uncond = v_b2[1:2]  # (1, 64)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    return guided_fn


def make_cfg_fn(diffusion_fn, neg_condition, cfg_scale: float):
    """Wrap a diffusion function with Classifier-Free Guidance.

    CFG doubles the diffusion head calls per step but is required for
    intelligible speech. Formula: v = v_uncond + scale * (v_cond - v_uncond)
    """
    if cfg_scale <= 1.0:
        return diffusion_fn

    def guided_fn(sample, timestep, condition):
        v_cond = diffusion_fn(sample, timestep, condition)
        v_uncond = diffusion_fn(sample, timestep, neg_condition)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    return guided_fn


def sample_latent(diffusion_fn, condition, solver: str = "dpm",
                  num_steps: int = 10, seed=None):
    """Dispatch to configured diffusion solver."""
    if solver == "dpm":
        return dpm_solver_2m_sample(diffusion_fn, condition, num_steps, seed)
    else:
        return ddpm_sample(diffusion_fn, condition, num_steps, seed)
