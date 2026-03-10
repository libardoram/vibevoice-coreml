"""MLX backend for component verification."""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from bench_mlx_common import (
    AcousticConnector as MlxAcousticConnector,
    DiffusionHead as MlxDiffusionHead,
    LMHead as MlxLMHead,
    Qwen2Decoder as MlxQwen2Decoder,
    SemanticConnector as MlxSemanticConnector,
    VAEDecoder as MlxVAEDecoder,
    make_rope as mlx_make_rope,
    to_mx,
)
from verify_common import benchmark_mlx


def load_models(weights: dict, num_layers: int = 28,
                 num_q_heads: int = 12, num_kv_heads: int = 2,
                 head_dim: int = 128, hidden_size: int = 1536,
                 rms_norm_eps: float = 1e-6, dtype=mx.float16) -> dict:
    """Load all MLX models for verification."""
    return {
        "diff": MlxDiffusionHead(weights, dtype),
        "vae": MlxVAEDecoder(weights, dtype),
        "ac_conn": MlxAcousticConnector(weights, dtype),
        "sem_conn": MlxSemanticConnector(weights, dtype),
        "lm_head": MlxLMHead(weights, key="lm_head.weight" if "lm_head.weight" in weights else "model.language_model.embed_tokens.weight", dtype=dtype),
        "lm": MlxQwen2Decoder(weights, "model.language_model.", num_layers,
                               num_q_heads, num_kv_heads, head_dim, hidden_size,
                               rms_norm_eps, has_norm=True, dtype=dtype),
        "dtype": dtype,
    }


def test_diffusion(m, noisy_np, t_np, cond_np, warmup, iters):
    dtype = m["dtype"]
    noisy_mx = mx.array(noisy_np).astype(dtype)
    cond_mx = mx.array(cond_np).astype(dtype)
    t_mx = mx.array(t_np).astype(dtype)
    out = m["diff"](noisy_mx, t_mx, cond_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["diff"](noisy_mx, t_mx, cond_mx), warmup, iters)
    return np.array(out), lat


def test_ddpm_loop(m, noise, cond_np, timesteps, alphas, ddpm_step_fn, warmup, iters):
    dtype = m["dtype"]
    cond_mx = mx.array(cond_np).astype(dtype)

    def _loop():
        s = mx.array(noise).astype(dtype)
        for i, t in enumerate(timesteps):
            pred = m["diff"](s, mx.array([float(t)]).astype(dtype), cond_mx)
            mx.eval(pred)
            s_np = np.array(s).astype(np.float32)
            pred_np = np.array(pred).astype(np.float32)
            at = float(alphas[int(t)])
            ap = float(alphas[int(timesteps[i + 1])]) if i < len(timesteps) - 1 else 1.0
            s_np = ddpm_step_fn(s_np, pred_np, at, ap)
            s = mx.array(s_np).astype(dtype)
        mx.eval(s)
        return s

    out = _loop()
    lat = benchmark_mlx(_loop, warmup=3, iters=20)
    return np.array(out), lat


def test_vae(m, latent_np, warmup, iters):
    dtype = m["dtype"]
    latent_mx = mx.array(latent_np).astype(dtype)
    out = m["vae"](latent_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["vae"](latent_mx), warmup, iters)
    return np.array(out), lat


def test_acoustic_connector(m, lat_np, warmup, iters):
    dtype = m["dtype"]
    lat_mx = mx.array(lat_np).astype(dtype)
    out = m["ac_conn"](lat_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["ac_conn"](lat_mx), warmup, iters)
    return np.array(out), lat


def test_semantic_connector(m, feat_np, warmup, iters):
    dtype = m["dtype"]
    feat_mx = mx.array(feat_np).astype(dtype)
    out = m["sem_conn"](feat_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["sem_conn"](feat_mx), warmup, iters)
    return np.array(out), lat


def test_lm_head(m, hidden_np, warmup, iters):
    dtype = m["dtype"]
    hidden_mx = mx.array(hidden_np).astype(dtype)
    out = m["lm_head"](hidden_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["lm_head"](hidden_mx), warmup, iters)
    return np.array(out), lat


def test_lm_decoder(m, h_np, cos_np, sin_np, head_dim, warmup, iters):
    dtype = m["dtype"]
    from rope import ROPE_THETA
    h_mx = mx.array(h_np).astype(dtype)
    cos_mx, sin_mx = mlx_make_rope(mx.array(0.0), head_dim, ROPE_THETA, dtype)
    out = m["lm"](h_mx, cos_mx, sin_mx)
    mx.eval(out)
    lat = benchmark_mlx(lambda: m["lm"](h_mx, cos_mx, sin_mx), warmup, iters)
    return np.array(out), lat
