#!/usr/bin/env python3
"""Export VibeVoice-ASR-HF components to CoreML.

Produces CoreML packages for the audio preprocessing pipeline:
  1. acoustic_encoder.mlpackage   — σ-VAE encoder (~340M)
  2. semantic_encoder.mlpackage   — semantic tokenizer encoder (~340M)
  3. acoustic_connector.mlpackage — projects acoustic to LM space
  4. semantic_connector.mlpackage — projects semantic to LM space
  5. lm_head.mlpackage            — hidden_state -> token logits

The Qwen2 LLM backbone (8B) is exported separately with quantization.

Usage:
    uv run python convert_coreml.py --output-dir ./build/vibevoice-asr-hf
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import typer

DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
AUTHOR = "Fluid Inference"

# Architecture constants from config
VAE_DIM = 64
SEMANTIC_DIM = 128
HIDDEN_SIZE = 3584  # Qwen2 hidden
NUM_LAYERS = 28
VOCAB_SIZE = 152064
SAMPLE_RATE = 24000
FRAME_RATE = 7.5
# 60s segment = 1,440,000 samples
SEGMENT_SAMPLES = int(60 * SAMPLE_RATE)


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    compute_precision: Optional[ct.precision]


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(f"Unknown compute units '{name}'. Choose from: {', '.join(mapping)}")
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if not normalized:
        return None
    mapping = {"FLOAT32": ct.precision.FLOAT32, "FLOAT16": ct.precision.FLOAT16}
    if normalized not in mapping:
        raise typer.BadParameter(f"Unknown precision '{name}'. Choose from: {', '.join(mapping)}")
    return mapping[normalized]


def _coreml_convert(traced, inputs, outputs, settings, compute_units_override=None):
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
    return ct.convert(traced, **kwargs)


def _save_mlpackage(model, path: Path, description: str) -> None:
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path}")


def _load_model(model_id: str):
    typer.echo(f"Loading model: {model_id}")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model


def export_acoustic_encoder(model, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableAcousticEncoder

    typer.echo("Exporting acoustic encoder...")
    encoder = TraceableAcousticEncoder(model.model.acoustic_tokenizer).eval()

    # Trace with a 10-second chunk (representative, not max)
    audio = torch.randn(1, 1, SAMPLE_RATE * 10)
    with torch.no_grad():
        ref = encoder(audio)
        typer.echo(f"  10s audio -> {ref.shape} features")

    traced = torch.jit.trace(encoder, (audio,), strict=False)

    # Variable-length input: 1s to 60s
    inputs = [ct.TensorType(
        name="audio",
        shape=(1, 1, ct.RangeDim(SAMPLE_RATE, SEGMENT_SAMPLES)),
        dtype=np.float32,
    )]
    outputs = [ct.TensorType(name="acoustic_features", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice ASR acoustic encoder (60s max)")
    return {
        "path": path.name,
        "inputs": {"audio": [1, 1, "1s-60s @ 24kHz"]},
        "outputs": {"acoustic_features": [1, "T", VAE_DIM]},
    }


def export_semantic_encoder(model, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableSemanticEncoder

    typer.echo("Exporting semantic encoder...")
    encoder = TraceableSemanticEncoder(model.model.semantic_tokenizer).eval()

    audio = torch.randn(1, 1, SAMPLE_RATE * 10)
    with torch.no_grad():
        ref = encoder(audio)
        typer.echo(f"  10s audio -> {ref.shape} features")

    traced = torch.jit.trace(encoder, (audio,), strict=False)

    inputs = [ct.TensorType(
        name="audio",
        shape=(1, 1, ct.RangeDim(SAMPLE_RATE, SEGMENT_SAMPLES)),
        dtype=np.float32,
    )]
    outputs = [ct.TensorType(name="semantic_features", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice ASR semantic encoder (60s max)")
    return {
        "path": path.name,
        "inputs": {"audio": [1, 1, "1s-60s @ 24kHz"]},
        "outputs": {"semantic_features": [1, "T", SEMANTIC_DIM]},
    }


def export_connectors(model, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableAcousticConnector, TraceableSemanticConnector

    results = {}

    typer.echo("Exporting acoustic connector...")
    ac = TraceableAcousticConnector(model.model.acoustic_connector).eval()
    feat = torch.randn(1, 75, VAE_DIM)  # 10s worth of frames
    with torch.no_grad():
        traced = torch.jit.trace(ac, (feat,), strict=False)
    inputs = [ct.TensorType(name="acoustic_features", shape=(1, ct.RangeDim(1, 450), VAE_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]
    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_connector.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice ASR acoustic connector")
    results["acoustic_connector"] = {
        "path": path.name,
        "inputs": {"acoustic_features": [1, "T", VAE_DIM]},
        "outputs": {"embedding": [1, "T", HIDDEN_SIZE]},
    }

    typer.echo("Exporting semantic connector...")
    sc = TraceableSemanticConnector(model.model.semantic_connector).eval()
    feat = torch.randn(1, 75, SEMANTIC_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(sc, (feat,), strict=False)
    inputs = [ct.TensorType(name="semantic_features", shape=(1, ct.RangeDim(1, 450), SEMANTIC_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]
    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_connector.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice ASR semantic connector")
    results["semantic_connector"] = {
        "path": path.name,
        "inputs": {"semantic_features": [1, "T", SEMANTIC_DIM]},
        "outputs": {"embedding": [1, "T", HIDDEN_SIZE]},
    }

    return results


def export_lm_head(model, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableLMHead

    typer.echo("Exporting LM head...")
    head = TraceableLMHead(model.lm_head).eval()

    hidden = torch.randn(1, 1, HIDDEN_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(head, (hidden,), strict=False)

    inputs = [ct.TensorType(name="hidden_state", shape=(1, 1, HIDDEN_SIZE), dtype=np.float32)]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "lm_head.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice ASR LM head")
    return {
        "path": path.name,
        "inputs": {"hidden_state": [1, 1, HIDDEN_SIZE]},
        "outputs": {"logits": [1, 1, VOCAB_SIZE]},
    }


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id", help="HuggingFace model ID"),
    output_dir: Path = typer.Option(Path("build/vibevoice-asr-hf"), "--output-dir"),
    compute_units: str = typer.Option("ALL", "--compute-units"),
    compute_precision: Optional[str] = typer.Option("FLOAT16", "--compute-precision"),
    skip_encoders: bool = typer.Option(False, "--skip-encoders", help="Skip tokenizer encoders"),
) -> None:
    """Export VibeVoice-ASR-HF audio preprocessing components to CoreML."""
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=_parse_compute_units(compute_units),
        compute_precision=_parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_id)

    components = {}

    if not skip_encoders:
        components["acoustic_encoder"] = export_acoustic_encoder(model, settings)
        components["semantic_encoder"] = export_semantic_encoder(model, settings)

    connector_results = export_connectors(model, settings)
    components.update(connector_results)

    components["lm_head"] = export_lm_head(model, settings)

    metadata = {
        "model_id": model_id,
        "model_type": "vibevoice_asr",
        "sample_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "vae_dim": VAE_DIM,
        "semantic_dim": SEMANTIC_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "vocab_size": VOCAB_SIZE,
        "max_audio_minutes": 60,
        "segment_samples": SEGMENT_SAMPLES,
        "languages": "50+",
        "coreml": {
            "compute_units": settings.compute_units.name,
            "compute_precision": (
                settings.compute_precision.name if settings.compute_precision else "FLOAT16"
            ),
            "minimum_deployment_target": "iOS17",
        },
        "components": components,
    }

    metadata_path = settings.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    typer.echo(f"\nExport complete. Metadata: {metadata_path}")
    typer.echo(f"Components: {list(components.keys())}")
    typer.echo("\nNote: Qwen2 LLM backbone (8B) export with INT4 quantization will be added separately.")


if __name__ == "__main__":
    app()
