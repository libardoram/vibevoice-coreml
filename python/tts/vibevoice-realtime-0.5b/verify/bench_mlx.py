#!/usr/bin/env python3
"""Benchmark VibeVoice-Realtime-0.5B components in MLX for comparison with CoreML.

Usage:
    uv run python bench_mlx.py
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from bench_mlx_common import (
    to_mx, benchmark, make_rope, rms_norm,
    DiffusionHead, VAEDecoder, AcousticConnector, Qwen2Decoder,
)

MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4/model.safetensors"

HIDDEN_SIZE = 896
VAE_DIM = 64
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
GQA_REPEAT = 7
INTERMEDIATE_SIZE = 4864
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1e6


# ─── EOS Classifier (0.5B only) ──────────────────────────────────────────────

class EOSClassifier:
    def __init__(self, weights, dtype=mx.float16):
        self.fc1_w = to_mx(weights["tts_eos_classifier.fc1.weight"], dtype)
        self.fc1_b = to_mx(weights["tts_eos_classifier.fc1.bias"], dtype)
        self.fc2_w = to_mx(weights["tts_eos_classifier.fc2.weight"], dtype)
        self.fc2_b = to_mx(weights["tts_eos_classifier.fc2.bias"], dtype)

    def __call__(self, x):
        x = nn.relu(x @ self.fc1_w.T + self.fc1_b)
        return mx.sigmoid(x @ self.fc2_w.T + self.fc2_b)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"MLX device: {mx.default_device()}")
    print(f"Loading weights from {MODEL_PATH}...")
    weights = load_file(str(MODEL_PATH))

    dtype = mx.float16
    print(f"Dtype: {dtype}")
    print()
    print("=" * 60)
    print("MLX BENCHMARK (float16, GPU)")
    print("=" * 60)

    # --- Diffusion Head ---
    print("\nDiffusion Head:")
    diff_head = DiffusionHead(weights, dtype)
    noisy = mx.random.normal((1, VAE_DIM)).astype(dtype)
    t_val = mx.array([500.0]).astype(dtype)
    cond = mx.random.normal((1, HIDDEN_SIZE)).astype(dtype)

    def run_diff():
        return diff_head(noisy, t_val, cond)
    benchmark(run_diff, label="single step")

    # Full 20-step DDPM loop
    def run_ddpm_loop():
        steps = 1000
        ac = []
        for i in range(steps):
            t = i / steps
            ac.append(math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        ac0 = ac[0]
        ac = [a / ac0 for a in ac]
        timesteps = list(range(0, steps, steps // 20))[::-1]
        s = mx.random.normal((1, VAE_DIM)).astype(dtype)
        c = cond
        for i, ts in enumerate(timesteps):
            pred = diff_head(s, mx.array([float(ts)]).astype(dtype), c)
            at = ac[ts]
            sa = math.sqrt(at)
            s1ma = math.sqrt(max(1 - at, 1e-8))
            px0 = sa * s - s1ma * pred
            if i < len(timesteps) - 1:
                ap = ac[timesteps[i + 1]]
            else:
                ap = 1.0
            pe = (s - sa * px0) / s1ma
            s = math.sqrt(ap) * px0 + math.sqrt(max(1 - ap, 1e-8)) * pe
        return s
    benchmark(run_ddpm_loop, warmup=5, iterations=50, label="20-step DDPM")

    # --- VAE Decoder ---
    print("\nVAE Decoder:")
    vae = VAEDecoder(weights, dtype)
    latent = mx.random.normal((1, VAE_DIM, 1)).astype(dtype)

    def run_vae():
        return vae(latent)
    out = run_vae()
    mx.eval(out)
    print(f"  Output shape: {out.shape}")
    benchmark(run_vae, label="single frame")

    # --- EOS Classifier ---
    print("\nEOS Classifier:")
    eos = EOSClassifier(weights, dtype)
    hidden = mx.random.normal((1, HIDDEN_SIZE)).astype(dtype)
    benchmark(lambda: eos(hidden), label="inference")

    # --- Acoustic Connector ---
    print("\nAcoustic Connector:")
    conn = AcousticConnector(weights, dtype)
    lat = mx.random.normal((1, 1, VAE_DIM)).astype(dtype)
    benchmark(lambda: conn(lat), label="inference")

    # --- Base LM (4 layers) ---
    print("\nBase LM (4 layers):")
    base_lm = Qwen2Decoder(weights, "model.language_model.", 4,
                            NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
                            RMS_NORM_EPS, has_norm=False, dtype=dtype)
    h = mx.random.normal((1, 1, HIDDEN_SIZE)).astype(dtype)
    cos, sin = make_rope(mx.array(0.0), HEAD_DIM, ROPE_THETA, dtype)
    benchmark(lambda: base_lm(h, cos, sin), label="decode")

    # Prefill 5 tokens
    h5 = mx.random.normal((1, 5, HIDDEN_SIZE)).astype(dtype)
    inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, HEAD_DIM, 2, dtype=mx.float32) / HEAD_DIM))
    pos5 = mx.arange(5, dtype=mx.float32)
    freqs5 = pos5[:, None] * inv_freq[None, :]
    freqs5 = mx.concatenate([freqs5, freqs5], axis=-1)
    cos5 = mx.cos(freqs5).reshape(1, 1, 5, HEAD_DIM).astype(dtype)
    sin5 = mx.sin(freqs5).reshape(1, 1, 5, HEAD_DIM).astype(dtype)
    benchmark(lambda: base_lm(h5, cos5, sin5), warmup=5, iterations=30, label="prefill(5)")

    # --- TTS LM (20 layers) ---
    print("\nTTS LM (20 layers):")
    tts_lm = Qwen2Decoder(weights, "model.tts_language_model.", 20,
                           NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
                           RMS_NORM_EPS, has_norm=True, dtype=dtype)
    benchmark(lambda: tts_lm(h, cos, sin), label="decode")
    benchmark(lambda: tts_lm(h5, cos5, sin5), warmup=5, iterations=30, label="prefill(5)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PER-FRAME ESTIMATE")
    print("=" * 60)
    tts_stats = benchmark(lambda: tts_lm(h, cos, sin), warmup=3, iterations=30, label="TTS LM decode")
    ddpm_stats = benchmark(run_ddpm_loop, warmup=3, iterations=20, label="DDPM 20 steps")
    vae_stats = benchmark(run_vae, warmup=3, iterations=30, label="VAE decode")

    total = tts_stats["median_ms"] + ddpm_stats["median_ms"] + vae_stats["median_ms"]
    audio_ms = 3200 / 24000 * 1000
    rtf = audio_ms / total if total > 0 else 0
    print(f"\n  Total: {total:.1f}ms for {audio_ms:.1f}ms audio")
    print(f"  Real-time factor: {rtf:.2f}x")


if __name__ == "__main__":
    main()
