"""Shared verification and benchmarking utilities for VibeVoice TTS models."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def benchmark(fn, warmup: int = 5, iters: int = 50) -> dict:
    """Run fn repeatedly and return timing statistics."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "min_ms": times[0],
    }


def benchmark_mlx(fn, warmup: int = 5, iters: int = 50) -> dict:
    """Benchmark with mx.eval sync after each call."""
    import mlx.core as mx
    for _ in range(warmup):
        fn()
        mx.eval(mx.zeros(1))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.eval(mx.zeros(1))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "min_ms": times[0],
    }


def compare(name: str, outputs: dict, threshold: float = 0.02) -> bool:
    """Compare outputs from multiple backends pairwise.

    Args:
        name: component name for display
        outputs: dict of {backend_name: np.ndarray}
        threshold: max absolute diff for PASS

    Returns True if all pairwise comparisons pass.
    """
    backends = list(outputs.keys())
    arrays = {k: np.array(v).flatten().astype(np.float32) for k, v in outputs.items()}
    n = min(len(a) for a in arrays.values())
    arrays = {k: v[:n] for k, v in arrays.items()}

    all_ok = True
    for i, b1 in enumerate(backends):
        for b2 in backends[i + 1:]:
            diff = np.abs(arrays[b1] - arrays[b2])
            ok = bool(diff.max() < threshold)
            if not ok:
                all_ok = False
            l1 = b1[:6].upper().ljust(6)
            l2 = b2[:6].upper().ljust(6)
            print(f"  {l1} vs {l2}: max={diff.max():.2e} mean={diff.mean():.2e}  {'PASS' if ok else 'FAIL'}")
    return all_ok


def format_latency(stats: dict) -> str:
    """Format benchmark stats as a short string."""
    return f"{stats['median_ms']:.2f}ms median"


def print_latency_row(name: str, latencies: dict):
    """Print a latency comparison row for multiple backends."""
    parts = [f"{k}: {v['median_ms']:.2f}ms" for k, v in latencies.items()]
    print(f"  Latency: {', '.join(parts)}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def pt_qwen2_forward(layers, norm, apply_norm: bool, h_np, cos_np, sin_np,
                      hidden_size: int, num_q_heads: int, num_kv_heads: int,
                      head_dim: int):
    """Run single-token Qwen2 forward pass through PyTorch layers (no KV cache)."""
    gqa_repeat = num_q_heads // num_kv_heads
    h = torch.from_numpy(h_np)
    cos_t = torch.from_numpy(cos_np).unsqueeze(1)
    sin_t = torch.from_numpy(sin_np).unsqueeze(1)
    scale = 1.0 / math.sqrt(head_dim)
    with torch.no_grad():
        for layer in layers:
            res = h
            h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(h).view(1, 1, num_q_heads, head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            q = q * cos_t + rotate_half(q) * sin_t
            k = k * cos_t + rotate_half(k) * sin_t
            k = k[:, :, None, :, :].expand(1, num_kv_heads, gqa_repeat, 1, head_dim) \
                .reshape(1, num_q_heads, 1, head_dim)
            v = v[:, :, None, :, :].expand(1, num_kv_heads, gqa_repeat, 1, head_dim) \
                .reshape(1, num_q_heads, 1, head_dim)
            attn = F.softmax(torch.matmul(q, k.transpose(2, 3)) * scale, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(1, 1, hidden_size)
            h = res + layer.self_attn.o_proj(out)
            res = h
            h = layer.post_attention_layernorm(h)
            h = res + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h))
        if apply_norm:
            h = norm(h)
    return h.numpy()


def print_summary_table(perf: dict, backends: list[str],
                        warmup: int, iters: int):
    """Print a performance summary table."""
    labels = [b.upper()[:6] for b in backends]
    header = "  " + f"{'Component':<25s}" + "".join(f" {l:>8s}" for l in labels)
    sep = "  " + "-" * 25 + (" " + "-" * 8) * len(labels)
    print(header)
    print(sep)
    for name, lat_dict in perf.items():
        row = f"  {name:<25s}"
        for b in backends:
            if b in lat_dict:
                row += f" {lat_dict[b]['median_ms']:>7.2f} "
            else:
                row += f" {'N/A':>7s} "
        print(row)
