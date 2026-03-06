#!/usr/bin/env python3
"""Export VibeVoice-1.5B TTS components to CoreML.

Produces CoreML packages for each component:
  1. diffusion_head.mlpackage     — single DDPM denoising step (~123M)
  2. vae_decoder.mlpackage        — acoustic σ-VAE decoder (~340M)
  3. vae_encoder.mlpackage        — acoustic σ-VAE encoder (~340M, for voice cloning)
  4. semantic_encoder.mlpackage   — semantic tokenizer encoder (~340M)
  5. acoustic_connector.mlpackage — latent-to-embedding projection
  6. semantic_connector.mlpackage — semantic-to-embedding projection
  7. lm_head.mlpackage            — next-token logits

The LLM backbone (Qwen2.5-1.5B) requires stateful KV cache handling
and is exported separately.

Usage:
    uv run python convert_coreml.py --output-dir ./build/vibevoice-1.5b
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

DEFAULT_MODEL_ID = "microsoft/VibeVoice-1.5B"
AUTHOR = "Fluid Inference"

# Architecture constants — will be read from config at runtime
CONFIGS = {
    "microsoft/VibeVoice-1.5B": {
        "vae_dim": 64,
        "semantic_dim": 128,
        "hidden_size": 1536,
        "num_layers": 28,
        "num_kv_heads": 2,
        "head_dim": 64,
        "ddpm_inference_steps": 20,
        "vocab_size": 151936,
    },
    "vibevoice/VibeVoice-7B": {
        "vae_dim": 64,
        "semantic_dim": 128,
        "hidden_size": 3584,
        "num_layers": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "ddpm_inference_steps": 20,
        "vocab_size": 152064,
    },
}

SAMPLE_RATE = 24000
FRAME_RATE = 7.5


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


def _get_config(model_id: str) -> dict:
    """Get architecture config, falling back to 1.5B defaults."""
    if model_id in CONFIGS:
        return CONFIGS[model_id]
    typer.echo(f"  Warning: no preset config for {model_id}, using 1.5B defaults")
    return CONFIGS["microsoft/VibeVoice-1.5B"]


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


def export_diffusion_head(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableDiffusionHead

    typer.echo("Exporting diffusion head...")
    head = TraceableDiffusionHead(model.model.prediction_head).eval()

    noisy = torch.randn(1, cfg["vae_dim"])
    timestep = torch.tensor([500])
    condition = torch.randn(1, cfg["hidden_size"])

    with torch.no_grad():
        traced = torch.jit.trace(head, (noisy, timestep, condition), strict=False)

    inputs = [
        ct.TensorType(name="noisy_latent", shape=(1, cfg["vae_dim"]), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(1,), dtype=np.int32),
        ct.TensorType(name="condition", shape=(1, cfg["hidden_size"]), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="predicted_noise", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_head.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice diffusion head (single DDPM step)")
    return {
        "path": path.name,
        "inputs": {"noisy_latent": [1, cfg["vae_dim"]], "timestep": [1], "condition": [1, cfg["hidden_size"]]},
        "outputs": {"predicted_noise": [1, cfg["vae_dim"]]},
    }


def export_vae_decoder(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableVAEDecoder

    typer.echo("Exporting VAE decoder...")
    decoder = TraceableVAEDecoder(model.model.acoustic_tokenizer.decoder).eval()

    latent = torch.randn(1, 1, cfg["vae_dim"])
    with torch.no_grad():
        ref = decoder(latent)
        frame_samples = ref.shape[-1]
    traced = torch.jit.trace(decoder, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, 1, cfg["vae_dim"]), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_decoder.mlpackage"
    _save_mlpackage(coreml_model, path, f"VibeVoice acoustic VAE decoder ({frame_samples} samples/frame)")
    return {
        "path": path.name,
        "frame_samples": int(frame_samples),
        "inputs": {"latent": [1, 1, cfg["vae_dim"]]},
        "outputs": {"audio": [1, 1, int(frame_samples)]},
    }


def export_vae_encoder(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableVAEEncoder

    typer.echo("Exporting VAE encoder...")
    encoder = TraceableVAEEncoder(model.model.acoustic_tokenizer.encoder).eval()

    # 1 second of audio at 24kHz
    audio = torch.randn(1, 1, SAMPLE_RATE)
    with torch.no_grad():
        ref = encoder(audio)
    traced = torch.jit.trace(encoder, (audio,), strict=False)

    # Use flexible length for audio input
    inputs = [ct.TensorType(name="audio", shape=(1, 1, ct.RangeDim(3200, SAMPLE_RATE * 30)), dtype=np.float32)]
    outputs = [ct.TensorType(name="mean", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice acoustic VAE encoder")
    return {
        "path": path.name,
        "inputs": {"audio": [1, 1, "variable"]},
        "outputs": {"mean": [1, "T", cfg["vae_dim"]]},
    }


def export_semantic_encoder(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableSemanticEncoder

    typer.echo("Exporting semantic encoder...")
    encoder = TraceableSemanticEncoder(model.model.semantic_tokenizer.encoder).eval()

    audio = torch.randn(1, 1, SAMPLE_RATE)
    with torch.no_grad():
        ref = encoder(audio)
    traced = torch.jit.trace(encoder, (audio,), strict=False)

    inputs = [ct.TensorType(name="audio", shape=(1, 1, ct.RangeDim(3200, SAMPLE_RATE * 30)), dtype=np.float32)]
    outputs = [ct.TensorType(name="features", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice semantic tokenizer encoder")
    return {
        "path": path.name,
        "inputs": {"audio": [1, 1, "variable"]},
        "outputs": {"features": [1, "T", cfg["semantic_dim"]]},
    }


def export_connectors(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableAcousticConnector, TraceableSemanticConnector

    results = {}

    typer.echo("Exporting acoustic connector...")
    ac = TraceableAcousticConnector(model.model.acoustic_connector).eval()
    latent = torch.randn(1, 1, cfg["vae_dim"])
    with torch.no_grad():
        traced = torch.jit.trace(ac, (latent,), strict=False)
    inputs = [ct.TensorType(name="speech_latent", shape=(1, 1, cfg["vae_dim"]), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]
    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_connector.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice acoustic connector")
    results["acoustic_connector"] = {
        "path": path.name,
        "inputs": {"speech_latent": [1, 1, cfg["vae_dim"]]},
        "outputs": {"embedding": [1, 1, cfg["hidden_size"]]},
    }

    typer.echo("Exporting semantic connector...")
    sc = TraceableSemanticConnector(model.model.semantic_connector).eval()
    feat = torch.randn(1, 1, cfg["semantic_dim"])
    with torch.no_grad():
        traced = torch.jit.trace(sc, (feat,), strict=False)
    inputs = [ct.TensorType(name="semantic_features", shape=(1, 1, cfg["semantic_dim"]), dtype=np.float32)]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]
    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_connector.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice semantic connector")
    results["semantic_connector"] = {
        "path": path.name,
        "inputs": {"semantic_features": [1, 1, cfg["semantic_dim"]]},
        "outputs": {"embedding": [1, 1, cfg["hidden_size"]]},
    }

    return results


def export_lm_head(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableLMHead

    typer.echo("Exporting LM head...")
    head = TraceableLMHead(model.lm_head).eval()

    hidden = torch.randn(1, 1, cfg["hidden_size"])
    with torch.no_grad():
        traced = torch.jit.trace(head, (hidden,), strict=False)

    inputs = [ct.TensorType(name="hidden_state", shape=(1, 1, cfg["hidden_size"]), dtype=np.float32)]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]

    coreml_model = _coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "lm_head.mlpackage"
    _save_mlpackage(coreml_model, path, "VibeVoice LM head")
    return {
        "path": path.name,
        "inputs": {"hidden_state": [1, 1, cfg["hidden_size"]]},
        "outputs": {"logits": [1, 1, cfg["vocab_size"]]},
    }


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID (supports VibeVoice-1.5B and VibeVoice-7B)",
    ),
    output_dir: Path = typer.Option(
        Path("build/vibevoice-1.5b"),
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
    skip_vae: bool = typer.Option(False, "--skip-vae", help="Skip VAE encoder/decoder"),
    skip_semantic: bool = typer.Option(False, "--skip-semantic", help="Skip semantic encoder"),
    skip_diffusion: bool = typer.Option(False, "--skip-diffusion", help="Skip diffusion head"),
) -> None:
    """Export VibeVoice 1.5B/7B TTS components to CoreML."""
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=_parse_compute_units(compute_units),
        compute_precision=_parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _get_config(model_id)
    model = _load_model(model_id)

    components = {}

    if not skip_diffusion:
        components["diffusion_head"] = export_diffusion_head(model, cfg, settings)

    if not skip_vae:
        components["vae_decoder"] = export_vae_decoder(model, cfg, settings)
        components["vae_encoder"] = export_vae_encoder(model, cfg, settings)

    if not skip_semantic:
        components["semantic_encoder"] = export_semantic_encoder(model, cfg, settings)

    connector_results = export_connectors(model, cfg, settings)
    components.update(connector_results)

    components["lm_head"] = export_lm_head(model, cfg, settings)

    metadata = {
        "model_id": model_id,
        "model_type": "vibevoice",
        "sample_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "vae_dim": cfg["vae_dim"],
        "semantic_dim": cfg["semantic_dim"],
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "vocab_size": cfg["vocab_size"],
        "ddpm_inference_steps": cfg["ddpm_inference_steps"],
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
    typer.echo("\nNote: LLM backbone (Qwen2.5) export with KV cache will be added separately.")


if __name__ == "__main__":
    app()
