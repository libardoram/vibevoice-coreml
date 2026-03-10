"""Shared RoPE computation for VibeVoice TTS CoreML pipelines."""

from __future__ import annotations

import numpy as np

ROPE_THETA = 1e6


def compute_rope_np(position: int, head_dim: int) -> tuple:
    """Compute cos/sin for RoPE at a single position. Returns (1, 1, head_dim)."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = float(position) * inv_freq
    freqs = np.concatenate([freqs, freqs])
    cos_np = np.cos(freqs)[None, None, :].astype(np.float32)
    sin_np = np.sin(freqs)[None, None, :].astype(np.float32)
    return cos_np, sin_np


def compute_rope_np_multi(positions, head_dim: int) -> tuple:
    """Compute RoPE for multiple positions at once. Returns (1, Q, head_dim)."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.array(list(positions), dtype=np.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    freqs = np.concatenate([freqs, freqs], axis=-1)
    cos_np = np.cos(freqs)[None, :, :].astype(np.float32)
    sin_np = np.sin(freqs)[None, :, :].astype(np.float32)
    return cos_np, sin_np
