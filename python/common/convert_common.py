"""Shared CoreML conversion utilities for VibeVoice TTS models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import typer


AUTHOR = "VibeVoice CoreML"


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    compute_precision: Optional[ct.precision]


def parse_compute_units(name: str) -> ct.ComputeUnit:
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: {', '.join(mapping)}"
        )
    return mapping[normalized]


def parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if not normalized:
        return None
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute precision '{name}'. Choose from: {', '.join(mapping)}"
        )
    return mapping[normalized]


def coreml_convert(traced, inputs, outputs, settings: ExportSettings,
                   compute_units_override=None, **extra_kwargs):
    """Convert a traced module to CoreML MLModel."""
    cu = compute_units_override or settings.compute_units
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
        "minimum_deployment_target": ct.target.iOS17,
    }
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    kwargs.update(extra_kwargs)
    return ct.convert(traced, **kwargs)


def save_mlpackage(model, path: Path, description: str) -> None:
    """Save a CoreML model with metadata."""
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path} ({description})")


def build_kv_state_specs(num_layers: int, num_kv_heads: int,
                         max_seq_len: int, head_dim: int) -> list:
    """Build ct.StateType list for KV cache buffers."""
    states = []
    for i in range(num_layers):
        for prefix in ("k", "v"):
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, num_kv_heads, max_seq_len, head_dim),
                    dtype=np.float16,
                ),
                name=f"{prefix}_cache_{i}",
            ))
    return states


def build_vae_state_specs(cache_layers) -> list:
    """Build ct.StateType list for streaming VAE decoder cache buffers."""
    states = []
    for name, _, ch, ctx, _ in cache_layers:
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, ch, ctx), dtype=np.float16),
            name=f"cache_{name}",
        ))
    return states


# ─── Torch utilities ─────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )
