#!/usr/bin/env python3
"""Export VibeVoice-Realtime-0.5B TTS components to CoreML.

Produces four CoreML packages:
  1. diffusion_head.mlpackage   — single DDPM denoising step (~40M)
  2. vae_decoder.mlpackage      — acoustic σ-VAE decoder (~340M)
  3. eos_classifier.mlpackage   — binary end-of-speech detector
  4. acoustic_connector.mlpackage — latent-to-embedding projection

The LLM backbone (Qwen2.5-0.5B, split into base LM + TTS LM) requires
stateful KV cache handling and is exported separately in a follow-up.

Usage:
    uv run python convert_coreml.py --output-dir ./build/vibevoice-realtime-0.5b
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

DEFAULT_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
AUTHOR = "Fluid Inference"

# Architecture constants from config.json
VAE_DIM = 64
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64  # hidden_size / num_attention_heads = 896 / 14
DDPM_INFERENCE_STEPS = 20
SAMPLE_RATE = 24000
FRAME_RATE = 7.5  # Hz — samples per frame = 24000 / 7.5 = 3200


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
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: {', '.join(mapping)}"
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if normalized == "":
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


def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs: list,
    outputs: list,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
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


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path} ({description})")


def _load_model(model_id: str):
    """Load VibeVoice Streaming model and its acoustic tokenizer."""
    typer.echo(f"Loading model: {model_id}")

    # Import here so the top-level module doesn't require transformers
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "vendor" / "VibeVoice"))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    return model


def export_diffusion_head(model, settings: ExportSettings) -> dict:
    """Export the diffusion head — single denoising step."""
    from traceable_modules import TraceableDiffusionHead

    typer.echo("Exporting diffusion head...")
    head = TraceableDiffusionHead(model.model.prediction_head).eval()

    # Trace with batch=1
    noisy = torch.randn(1, VAE_DIM)
    timestep = torch.tensor([500])
    condition = torch.randn(1, HIDDEN_SIZE)

    with torch.no_grad():
        traced = torch.jit.trace(head, (noisy, timestep, condition), strict=False)

    inputs = [
        ct.TensorType(name="noisy_latent", shape=(1, VAE_DIM), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(1,), dtype=np.int32),
        ct.TensorType(name="condition", shape=(1, HIDDEN_SIZE), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="predicted_noise", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_head.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice diffusion head (single DDPM step)")

    return {
        "path": path.name,
        "inputs": {"noisy_latent": [1, VAE_DIM], "timestep": [1], "condition": [1, HIDDEN_SIZE]},
        "outputs": {"predicted_noise": [1, VAE_DIM]},
    }


def export_vae_decoder(model, settings: ExportSettings) -> dict:
    """Export the acoustic VAE decoder."""
    from traceable_modules import TraceableVAEDecoder

    typer.echo("Exporting VAE decoder...")
    tokenizer = model.model.acoustic_tokenizer
    if tokenizer is None:
        typer.echo("  WARNING: acoustic_tokenizer not loaded in model, attempting to load separately")
        from transformers import AutoModel
        tokenizer = AutoModel.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            subfolder="acoustic_tokenizer",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    decoder = TraceableVAEDecoder(tokenizer.decoder).eval()

    # One frame of latent -> audio
    latent = torch.randn(1, 1, VAE_DIM)
    with torch.no_grad():
        ref_audio = decoder(latent)
        frame_samples = ref_audio.shape[-1]

    traced = torch.jit.trace(decoder, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, 1, VAE_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_decoder.mlpackage"
    _save_mlpackage(coreml_model, path, f"VibeVoice acoustic VAE decoder ({frame_samples} samples/frame)")

    return {
        "path": path.name,
        "frame_samples": int(frame_samples),
        "inputs": {"latent": [1, 1, VAE_DIM]},
        "outputs": {"audio": [1, 1, int(frame_samples)]},
    }


def export_eos_classifier(model, settings: ExportSettings) -> dict:
    """Export the EOS binary classifier."""
    from traceable_modules import TraceableEOSClassifier

    typer.echo("Exporting EOS classifier...")
    classifier = TraceableEOSClassifier(model.tts_eos_classifier).eval()

    hidden = torch.randn(1, HIDDEN_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(classifier, (hidden,), strict=False)

    inputs = [ct.TensorType(name="hidden_state", shape=(1, HIDDEN_SIZE), dtype=np.float32)]
    outputs = [ct.TensorType(name="eos_probability", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "eos_classifier.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice EOS classifier (sigmoid probability)")

    return {
        "path": path.name,
        "inputs": {"hidden_state": [1, HIDDEN_SIZE]},
        "outputs": {"eos_probability": [1, 1]},
    }


def export_acoustic_connector(model, settings: ExportSettings) -> dict:
    """Export the acoustic connector (latent -> LM embedding)."""
    from traceable_modules import TraceableAcousticConnector

    typer.echo("Exporting acoustic connector...")
    connector = TraceableAcousticConnector(model.model.acoustic_connector).eval()

    latent = torch.randn(1, 1, VAE_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(connector, (latent,), strict=False)

    inputs = [ct.TensorType(name="speech_latent", shape=(1, 1, VAE_DIM), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_connector.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice acoustic connector")

    return {
        "path": path.name,
        "inputs": {"speech_latent": [1, 1, VAE_DIM]},
        "outputs": {"embedding": [1, 1, HIDDEN_SIZE]},
    }


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID or local path",
    ),
    output_dir: Path = typer.Option(
        Path("build/vibevoice-realtime-0.5b"),
        "--output-dir",
        help="Directory for CoreML outputs",
    ),
    compute_units: str = typer.Option(
        "ALL",
        "--compute-units",
        help="CoreML compute units: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE",
    ),
    compute_precision: Optional[str] = typer.Option(
        "FLOAT16",
        "--compute-precision",
        help="Export precision: FLOAT32 or FLOAT16",
    ),
    skip_vae: bool = typer.Option(False, "--skip-vae", help="Skip VAE decoder export"),
    skip_diffusion: bool = typer.Option(False, "--skip-diffusion", help="Skip diffusion head export"),
) -> None:
    """Export VibeVoice-Realtime-0.5B components to CoreML."""
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=_parse_compute_units(compute_units),
        compute_precision=_parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_id)

    components = {}

    if not skip_diffusion:
        components["diffusion_head"] = export_diffusion_head(model, settings)

    if not skip_vae:
        components["vae_decoder"] = export_vae_decoder(model, settings)

    components["eos_classifier"] = export_eos_classifier(model, settings)
    components["acoustic_connector"] = export_acoustic_connector(model, settings)

    metadata = {
        "model_id": model_id,
        "model_type": "vibevoice_streaming",
        "sample_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "vae_dim": VAE_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "ddpm_inference_steps": DDPM_INFERENCE_STEPS,
        "text_window_size": 5,
        "speech_window_size": 6,
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
    typer.echo(f"Components exported: {list(components.keys())}")
    typer.echo("\nNote: LLM backbone (base LM + TTS LM) export requires stateful")
    typer.echo("KV cache handling and will be added in a follow-up conversion.")


if __name__ == "__main__":
    app()
