#!/usr/bin/env python3
"""Export VibeVoice multi-speaker TTS components to CoreML.

Produces CoreML packages for each component:
  1. diffusion_head.mlpackage          — single DDPM denoising step
  2. vae_decoder_streaming.mlpackage   — streaming σ-VAE decoder (stateful, conv caches)
  3. vae_encoder.mlpackage             — acoustic σ-VAE encoder (for voice cloning)
  4. acoustic_connector.mlpackage      — latent-to-embedding projection
  5. semantic_connector.mlpackage      — semantic-to-embedding projection
  6. lm_head.mlpackage                 — next-token logits

The LLM backbone (Qwen2.5) requires stateful KV cache handling
and is exported separately via convert_stateful_lm.py.

Usage:
    uv run python convert_coreml.py --model-id microsoft/VibeVoice-1.5B
    uv run python convert_coreml.py --model-id vibevoice/VibeVoice-7B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
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
    build_vae_state_specs,
)

_PARENT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIRS = {
    "microsoft/VibeVoice-1.5B": str(_PARENT / "build/vibevoice-1.5b"),
    "vibevoice/VibeVoice-7B": str(_PARENT / "build/vibevoice-7b"),
}

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


def _get_config(model_id: str) -> dict:
    """Get architecture config."""
    if model_id in CONFIGS:
        return CONFIGS[model_id]
    typer.echo(f"  Error: unknown model {model_id}. Supported: {list(CONFIGS.keys())}")
    raise typer.Exit(1)


def _load_model(model_id: str):
    typer.echo(f"Loading model: {model_id}")
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def export_diffusion_head(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableDiffusionHead

    typer.echo("Exporting diffusion head...")
    head = TraceableDiffusionHead(model.model.prediction_head).eval()

    noisy = torch.randn(1, cfg["vae_dim"])
    timestep = torch.tensor([500.0])
    condition = torch.randn(1, cfg["hidden_size"])

    with torch.no_grad():
        traced = torch.jit.trace(head, (noisy, timestep, condition), strict=False)

    inputs = [
        ct.TensorType(name="noisy_latent", shape=(1, cfg["vae_dim"]), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(1,), dtype=np.float32),
        ct.TensorType(name="condition", shape=(1, cfg["hidden_size"]), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="predicted_noise", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_head.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice diffusion head (single DDPM step)")
    return {
        "path": path.name,
        "inputs": {"noisy_latent": [1, cfg["vae_dim"]], "timestep": [1], "condition": [1, cfg["hidden_size"]]},
        "outputs": {"predicted_noise": [1, cfg["vae_dim"]]},
    }


def export_diffusion_loop(model, cfg: dict, settings: ExportSettings,
                          num_steps: int = 10, cfg_scale: float = 1.3) -> dict:
    from traceable_modules import TraceableDiffusionLoopCFG

    typer.echo(f"Exporting fused diffusion loop (DPM-Solver++ 2M, {num_steps} steps, runtime CFG)...")
    loop = TraceableDiffusionLoopCFG(
        model.model.prediction_head,
        num_steps=num_steps,
    ).eval()

    noise = torch.randn(1, cfg["vae_dim"])
    condition = torch.randn(1, cfg["hidden_size"])
    neg_condition = torch.randn(1, cfg["hidden_size"])
    cfg_scale_t = torch.tensor([cfg_scale])

    with torch.no_grad():
        traced = torch.jit.trace(loop, (noise, condition, neg_condition, cfg_scale_t), strict=False)

    inputs = [
        ct.TensorType(name="noise", shape=(1, cfg["vae_dim"]), dtype=np.float32),
        ct.TensorType(name="condition", shape=(1, cfg["hidden_size"]), dtype=np.float32),
        ct.TensorType(name="neg_condition", shape=(1, cfg["hidden_size"]), dtype=np.float32),
        ct.TensorType(name="cfg_scale", shape=(1,), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="latent", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_loop.mlpackage"
    save_mlpackage(coreml_model, path,
                   f"VibeVoice fused diffusion loop (DPM++ 2M, {num_steps}s)")
    return {
        "path": path.name,
        "inputs": {
            "noise": [1, cfg["vae_dim"]],
            "condition": [1, cfg["hidden_size"]],
            "neg_condition": [1, cfg["hidden_size"]],
            "cfg_scale": [1],
        },
        "outputs": {"latent": [1, cfg["vae_dim"]]},
    }


def export_vae_decoder_streaming(model, cfg: dict, settings: ExportSettings) -> dict:
    """Export a stateful streaming VAE decoder with conv cache state buffers.

    Each of the 34 causal conv layers maintains a small context buffer as
    ct.StateType, allowing frame-by-frame decode with full temporal context.
    Requires iOS 18+ for ct.StateType. Must load with CPU_AND_GPU (ANE fails).
    """
    from traceable_modules import TraceableStreamingVAEDecoder

    typer.echo("Exporting streaming VAE decoder (stateful, T=1)...")
    wrapper = TraceableStreamingVAEDecoder(model.model.acoustic_tokenizer.decoder).eval()

    latent = torch.randn(1, cfg["vae_dim"], 1)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, cfg["vae_dim"], 1), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]
    states = build_vae_state_specs(wrapper.cache_layers)

    coreml_model = coreml_convert(traced, inputs, outputs, settings,
                                   states=states,
                                   minimum_deployment_target=ct.target.iOS18)

    path = settings.output_dir / "vae_decoder_streaming.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice streaming VAE decoder (stateful, T=1)")

    return {
        "path": path.name,
        "num_cache_layers": len(wrapper.cache_layers),
        "inputs": {"latent": [1, cfg["vae_dim"], 1]},
        "outputs": {"audio": [1, 1, 3200]},
        "cache_layers": [
            {"name": name, "channels": ch, "context_size": ctx}
            for name, _, ch, ctx, _ in wrapper.cache_layers
        ],
    }


def export_vae_encoder(model, cfg: dict, settings: ExportSettings) -> dict:
    from traceable_modules import TraceableVAEEncoder

    typer.echo("Exporting VAE encoder...")
    encoder = TraceableVAEEncoder(model.model.acoustic_tokenizer.encoder).eval()

    # 10 seconds of audio at 24kHz (fixed length for voice cloning reference)
    audio = torch.randn(1, 1, SAMPLE_RATE * 10)
    with torch.no_grad():
        ref = encoder(audio)
    traced = torch.jit.trace(encoder, (audio,), strict=False)

    # Fixed length — dynamic padding in tokenizer is incompatible with RangeDim
    enc_samples = SAMPLE_RATE * 10  # 10 seconds for voice cloning reference
    inputs = [ct.TensorType(name="audio", shape=(1, 1, enc_samples), dtype=np.float32)]
    outputs = [ct.TensorType(name="latent", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_encoder.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice acoustic VAE encoder")
    return {
        "path": path.name,
        "inputs": {"audio": [1, 1, enc_samples]},
        "outputs": {"latent": [1, cfg["vae_dim"], "T"]},
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
    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_connector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice acoustic connector")
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
    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "semantic_connector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice semantic connector")
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

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "lm_head.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice LM head")
    return {
        "path": path.name,
        "inputs": {"hidden_state": [1, 1, cfg["hidden_size"]]},
        "outputs": {"logits": [1, 1, cfg["vocab_size"]]},
    }


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(
        ...,
        "--model-id",
        help="HuggingFace model ID (e.g. microsoft/VibeVoice-1.5B, vibevoice/VibeVoice-7B)",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        help="Directory for CoreML outputs (default: build/<model>)",
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
    skip_diffusion: bool = typer.Option(False, "--skip-diffusion", help="Skip diffusion head"),
    fused_diffusion: bool = typer.Option(False, "--fused-diffusion", help="Also export fused diffusion loop"),
    fused_diffusion_steps: int = typer.Option(10, "--fused-diffusion-steps", help="Steps for fused diffusion loop"),
    fused_diffusion_cfg: float = typer.Option(1.3, "--fused-diffusion-cfg", help="CFG scale for fused diffusion loop"),
) -> None:
    """Export VibeVoice multi-speaker TTS components to CoreML."""
    if output_dir is None:
        output_dir = Path(DEFAULT_BUILD_DIRS.get(model_id, str(_PARENT / f"build/{model_id.split('/')[-1].lower()}")))
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=parse_compute_units(compute_units),
        compute_precision=parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _get_config(model_id)
    model = _load_model(model_id)

    components = {}

    if not skip_diffusion:
        components["diffusion_head"] = export_diffusion_head(model, cfg, settings)

    if fused_diffusion:
        components["diffusion_loop"] = export_diffusion_loop(
            model, cfg, settings,
            num_steps=fused_diffusion_steps, cfg_scale=fused_diffusion_cfg,
        )

    if not skip_vae:
        components["vae_decoder_streaming"] = export_vae_decoder_streaming(model, cfg, settings)
        components["vae_encoder"] = export_vae_encoder(model, cfg, settings)

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
