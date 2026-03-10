#!/usr/bin/env python3
"""Export VibeVoice-ASR-HF components to CoreML.

Produces:
  1. acoustic_encoder.mlpackage     — σ-VAE encoder (24kHz audio → vae_dim=64 features)
  2. semantic_encoder.mlpackage     — σ-VAE encoder (24kHz audio → sem_dim=128 features)
  3. acoustic_projector.mlpackage   — Linear(64→3584) + RMSNorm + Linear(3584→3584)
  4. semantic_projector.mlpackage   — Linear(128→3584) + RMSNorm + Linear(3584→3584)
  5. lm_head.mlpackage              — Linear(3584→152064) for vocabulary projection
  6. embed_tokens.bin               — Token embeddings (152064, 3584) float16 binary

The LLM backbone (Qwen2-7B) is exported separately by convert_stateful_lm.py.

Usage:
    uv run python convert/convert_coreml.py --output-dir build/vibevoice-asr
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("coremltools").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import coremltools as ct
import numpy as np
import torch
import typer

from convert_common import (
    ExportSettings,
    parse_compute_units,
    parse_compute_precision,
    coreml_convert,
    save_mlpackage,
)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"

# Architecture constants
VAE_DIM = 64
SEM_DIM = 128
HIDDEN_SIZE = 3584
VOCAB_SIZE = 152064

# Encoder: 60s chunks at 24kHz = 1,440,000 samples, 3200x downsample → 450 tokens
CHUNK_SAMPLES = 1_440_000
CHUNK_TOKENS = CHUNK_SAMPLES // 3200  # 450

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def load_model():
    """Load VibeVoice-ASR-HF model from HuggingFace."""
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import AutoModelForSpeechSeq2Seq

    typer.echo(f"Loading model: {MODEL_ID}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=torch.float32,
    )
    model.eval()
    return model


class TraceableEncoder(torch.nn.Module):
    """Wrap σ-VAE encoder for tracing: audio (1, 1, samples) → features."""

    def __init__(self, encoder, is_acoustic: bool = True):
        super().__init__()
        self.encoder = encoder
        self.is_acoustic = is_acoustic

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        output = self.encoder(audio)
        return output.latents  # (1, T, dim)


class TraceableFusedEncoder(torch.nn.Module):
    """Wrap both σ-VAE encoders: audio (1, 1, samples) → (acoustic, semantic) features."""

    def __init__(self, acoustic_encoder, semantic_encoder):
        super().__init__()
        self.acoustic_encoder = acoustic_encoder
        self.semantic_encoder = semantic_encoder

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ac_output = self.acoustic_encoder(audio)
        sem_output = self.semantic_encoder(audio)
        return ac_output.latents, sem_output.latents


class TraceableProjector(torch.nn.Module):
    """Wrap multi_modal_projector path for tracing."""

    def __init__(self, linear1, norm, linear2):
        super().__init__()
        self.linear1 = linear1
        self.norm = norm
        self.linear2 = linear2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.linear1(features)
        x = self.norm(x)
        return self.linear2(x)


def export_acoustic_encoder(model, settings: ExportSettings) -> dict:
    """Export acoustic σ-VAE encoder."""
    typer.echo("Exporting acoustic encoder...")

    encoder = TraceableEncoder(model.acoustic_tokenizer_encoder, is_acoustic=True).eval()
    dummy = torch.randn(1, 1, CHUNK_SAMPLES)

    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy)

    inputs = [ct.TensorType(name="audio", shape=(1, 1, CHUNK_SAMPLES), dtype=np.float32)]
    outputs = [ct.TensorType(name="features", dtype=np.float32)]

    # CPU_AND_GPU avoids extremely slow ANE compilation for σ-VAE conv networks
    coreml_model = coreml_convert(traced, inputs, outputs, settings,
                                   compute_units_override=ct.ComputeUnit.CPU_AND_GPU)
    path = settings.output_dir / "acoustic_encoder.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR acoustic encoder (60s chunk)")

    # Verify output shape
    out = coreml_model.predict({"audio": np.random.randn(1, 1, CHUNK_SAMPLES).astype(np.float32)})
    typer.echo(f"  Output shape: {out['features'].shape}")

    return {"path": path.name, "inputs": {"audio": [1, 1, CHUNK_SAMPLES]},
            "outputs": {"features": list(out["features"].shape)}}


def export_semantic_encoder(model, settings: ExportSettings) -> dict:
    """Export semantic σ-VAE encoder."""
    typer.echo("Exporting semantic encoder...")

    encoder = TraceableEncoder(model.semantic_tokenizer_encoder, is_acoustic=False).eval()
    dummy = torch.randn(1, 1, CHUNK_SAMPLES)

    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy)

    inputs = [ct.TensorType(name="audio", shape=(1, 1, CHUNK_SAMPLES), dtype=np.float32)]
    outputs = [ct.TensorType(name="features", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings,
                                   compute_units_override=ct.ComputeUnit.CPU_AND_GPU)
    path = settings.output_dir / "semantic_encoder.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR semantic encoder (60s chunk)")

    out = coreml_model.predict({"audio": np.random.randn(1, 1, CHUNK_SAMPLES).astype(np.float32)})
    typer.echo(f"  Output shape: {out['features'].shape}")

    return {"path": path.name, "inputs": {"audio": [1, 1, CHUNK_SAMPLES]},
            "outputs": {"features": list(out["features"].shape)}}


def export_acoustic_projector(model, settings: ExportSettings) -> dict:
    """Export acoustic path of multi_modal_projector."""
    typer.echo("Exporting acoustic projector...")

    proj = model.multi_modal_projector
    traceable = TraceableProjector(
        proj.acoustic_linear_1, proj.acoustic_norm, proj.acoustic_linear_2
    ).eval()

    dummy = torch.randn(1, CHUNK_TOKENS, VAE_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(traceable, dummy)

    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=CHUNK_TOKENS, default=CHUNK_TOKENS)
    inputs = [ct.TensorType(name="features", shape=(1, seq_dim, VAE_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_projector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR acoustic projector (64→3584)")
    return {"path": path.name}


def export_semantic_projector(model, settings: ExportSettings) -> dict:
    """Export semantic path of multi_modal_projector."""
    typer.echo("Exporting semantic projector...")

    proj = model.multi_modal_projector
    traceable = TraceableProjector(
        proj.semantic_linear_1, proj.semantic_norm, proj.semantic_linear_2
    ).eval()

    dummy = torch.randn(1, CHUNK_TOKENS, SEM_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(traceable, dummy)

    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=CHUNK_TOKENS, default=CHUNK_TOKENS)
    inputs = [ct.TensorType(name="features", shape=(1, seq_dim, SEM_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_projector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR semantic projector (128→3584)")
    return {"path": path.name}


def export_fused_encoder(model, settings: ExportSettings) -> dict:
    """Export fused acoustic+semantic encoder (single model, two outputs)."""
    typer.echo("Exporting fused encoder...")

    encoder = TraceableFusedEncoder(
        model.acoustic_tokenizer_encoder, model.semantic_tokenizer_encoder,
    ).eval()
    dummy = torch.randn(1, 1, CHUNK_SAMPLES)

    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy)

    inputs = [ct.TensorType(name="audio", shape=(1, 1, CHUNK_SAMPLES), dtype=np.float32)]
    outputs = [
        ct.TensorType(name="acoustic_features", dtype=np.float32),
        ct.TensorType(name="semantic_features", dtype=np.float32),
    ]

    coreml_model = coreml_convert(traced, inputs, outputs, settings,
                                   compute_units_override=ct.ComputeUnit.CPU_AND_GPU)
    path = settings.output_dir / "fused_encoder.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR fused encoder (60s chunk → dim64+dim128)")

    # Verify
    out = coreml_model.predict({"audio": np.random.randn(1, 1, CHUNK_SAMPLES).astype(np.float32)})
    typer.echo(f"  Acoustic: {out['acoustic_features'].shape}, Semantic: {out['semantic_features'].shape}")

    return {"path": path.name}


def export_fused_projector(model, settings: ExportSettings) -> dict:
    """Export fused acoustic+semantic projector (single model, summed output)."""
    typer.echo("Exporting fused projector...")

    proj = model.multi_modal_projector

    class FusedProjector(torch.nn.Module):
        def __init__(self, ac_l1, ac_norm, ac_l2, sem_l1, sem_norm, sem_l2):
            super().__init__()
            self.ac_l1, self.ac_norm, self.ac_l2 = ac_l1, ac_norm, ac_l2
            self.sem_l1, self.sem_norm, self.sem_l2 = sem_l1, sem_norm, sem_l2

        def forward(self, acoustic: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
            ac = self.ac_l2(self.ac_norm(self.ac_l1(acoustic)))
            sem = self.sem_l2(self.sem_norm(self.sem_l1(semantic)))
            return ac + sem

    fused = FusedProjector(
        proj.acoustic_linear_1, proj.acoustic_norm, proj.acoustic_linear_2,
        proj.semantic_linear_1, proj.semantic_norm, proj.semantic_linear_2,
    ).eval()

    dummy_ac = torch.randn(1, CHUNK_TOKENS, VAE_DIM)
    dummy_sem = torch.randn(1, CHUNK_TOKENS, SEM_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(fused, (dummy_ac, dummy_sem))

    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=CHUNK_TOKENS * 10, default=CHUNK_TOKENS)
    inputs = [
        ct.TensorType(name="acoustic_features", shape=(1, seq_dim, VAE_DIM), dtype=np.float32),
        ct.TensorType(name="semantic_features", shape=(1, seq_dim, SEM_DIM), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "fused_projector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR fused projector (64+128→3584)")
    return {"path": path.name}


def export_lm_head(model, settings: ExportSettings) -> dict:
    """Export LM head (hidden→logits)."""
    from traceable_common import TraceableLMHead

    typer.echo("Exporting LM head...")
    head = TraceableLMHead(model.language_model.lm_head).eval()
    dummy = torch.randn(1, 1, HIDDEN_SIZE)

    with torch.no_grad():
        traced = torch.jit.trace(head, dummy)

    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=512, default=1)
    inputs = [ct.TensorType(name="hidden_state", shape=(1, seq_dim, HIDDEN_SIZE), dtype=np.float32)]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "lm_head.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice ASR LM head (3584→152064)")
    return {"path": path.name}


def export_embeddings(model, settings: ExportSettings) -> dict:
    """Export token embeddings as .bin (float16, with uint32 header).

    Binary format: [vocab_size: uint32] [hidden_size: uint32] [data: float16[vocab_size * hidden_size]]
    This is compact (half of float32) and trivially readable from Swift/Kotlin/Python
    without numpy dependency.
    """
    typer.echo("Exporting token embeddings...")
    embed = model.language_model.model.embed_tokens.weight.detach().float().numpy()
    embed_f16 = embed.astype(np.float16)

    path = settings.output_dir / "embed_tokens.bin"
    with open(path, "wb") as f:
        vocab_size, hidden_size = embed.shape
        f.write(np.array([vocab_size, hidden_size], dtype=np.uint32).tobytes())
        f.write(embed_f16.tobytes())

    size_mb = path.stat().st_size / 1e6
    f32_mb = embed.nbytes / 1e6
    typer.echo(f"  Saved: {path} ({embed.shape}, {size_mb:.1f}MB, was {f32_mb:.1f}MB as float32)")
    return {"path": path.name, "shape": list(embed.shape), "dtype": "float16"}


@app.command()
def main(
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    compute_units: str = typer.Option("ALL", "--compute-units"),
    compute_precision: str = typer.Option("FLOAT16", "--compute-precision"),
    skip_encoders: bool = typer.Option(False, "--skip-encoders"),
    skip_projectors: bool = typer.Option(False, "--skip-projectors"),
    fuse_encoders: bool = typer.Option(False, "--fuse-encoders",
                                        help="Export fused acoustic+semantic encoder"),
    fuse_projectors: bool = typer.Option(False, "--fuse-projectors",
                                          help="Export fused acoustic+semantic projector"),
):
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=parse_compute_units(compute_units),
        compute_precision=parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    components = {}

    if not skip_encoders:
        if fuse_encoders:
            components["fused_encoder"] = export_fused_encoder(model, settings)
        else:
            components["acoustic_encoder"] = export_acoustic_encoder(model, settings)
            components["semantic_encoder"] = export_semantic_encoder(model, settings)

    if not skip_projectors:
        if fuse_projectors:
            components["fused_projector"] = export_fused_projector(model, settings)
        else:
            components["acoustic_projector"] = export_acoustic_projector(model, settings)
            components["semantic_projector"] = export_semantic_projector(model, settings)

    components["lm_head"] = export_lm_head(model, settings)
    components["embed_tokens"] = export_embeddings(model, settings)

    import json
    meta_path = settings.output_dir / "metadata.json"
    json.dump(components, meta_path.open("w"), indent=2)
    typer.echo(f"\nExport complete. Metadata: {meta_path}")
    typer.echo(f"Components exported: {list(components.keys())}")


if __name__ == "__main__":
    app()
