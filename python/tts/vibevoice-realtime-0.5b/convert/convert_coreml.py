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
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("coremltools").setLevel(logging.ERROR)
from typing import Optional, Tuple

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

DEFAULT_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"

# Architecture constants from config.json
VAE_DIM = 64
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64  # hidden_size / num_attention_heads = 896 / 14
DDPM_INFERENCE_STEPS = 20
SAMPLE_RATE = 24000
FRAME_RATE = 7.5  # Hz — samples per frame = 24000 / 7.5 = 3200




def _load_model(model_id: str):
    """Load VibeVoice Streaming model and its acoustic tokenizer."""
    typer.echo(f"Loading model: {model_id}")

    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )

    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()

    return model


def export_diffusion_head(model, settings: ExportSettings) -> dict:
    """Export the diffusion head — single denoising step."""
    from traceable_modules import TraceableDiffusionHead

    typer.echo("Exporting diffusion head...")
    head = TraceableDiffusionHead(model.model.prediction_head).eval()

    # Trace with batch=1 (timestep as float — wrapper casts to long)
    noisy = torch.randn(1, VAE_DIM)
    timestep = torch.tensor([500.0])
    condition = torch.randn(1, HIDDEN_SIZE)

    with torch.no_grad():
        traced = torch.jit.trace(head, (noisy, timestep, condition), strict=False)

    inputs = [
        ct.TensorType(name="noisy_latent", shape=(1, VAE_DIM), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(1,), dtype=np.float32),
        ct.TensorType(name="condition", shape=(1, HIDDEN_SIZE), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="predicted_noise", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_head.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice diffusion head (single DDPM step)")

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

    # One frame of latent -> audio; decoder expects (B, C, T) layout
    latent = torch.randn(1, VAE_DIM, 1)
    with torch.no_grad():
        ref_audio = decoder(latent)
        frame_samples = ref_audio.shape[-1]

    traced = torch.jit.trace(decoder, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, VAE_DIM, 1), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_decoder.mlpackage"
    save_mlpackage(coreml_model, path, f"VibeVoice acoustic VAE decoder ({frame_samples} samples/frame)")

    return {
        "path": path.name,
        "frame_samples": int(frame_samples),
        "inputs": {"latent": [1, VAE_DIM, 1]},
        "outputs": {"audio": [1, 1, int(frame_samples)]},
    }


def export_vae_decoder_batch(model, settings: ExportSettings, max_frames: int = 256) -> dict:
    """Export a batch VAE decoder that processes multiple frames at once.

    The decoder is causal, so zero-padded frames at the end don't affect
    earlier frames. This gives the conv layers full temporal context across
    frames, matching streaming-with-cache quality.
    """
    from traceable_modules import TraceableVAEDecoder

    typer.echo(f"Exporting batch VAE decoder (T={max_frames})...")
    tokenizer = model.model.acoustic_tokenizer
    if tokenizer is None:
        typer.echo("  WARNING: acoustic_tokenizer not loaded, loading separately")
        from transformers import AutoModel
        tokenizer = AutoModel.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            subfolder="acoustic_tokenizer",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    decoder = TraceableVAEDecoder(tokenizer.decoder).eval()

    latent = torch.randn(1, VAE_DIM, max_frames)
    with torch.no_grad():
        ref_audio = decoder(latent)
        total_samples = ref_audio.shape[-1]
        samples_per_frame = total_samples // max_frames

    traced = torch.jit.trace(decoder, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, VAE_DIM, max_frames), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "vae_decoder_batch.mlpackage"
    save_mlpackage(coreml_model, path,
                    f"VibeVoice batch VAE decoder (T={max_frames}, {samples_per_frame} samples/frame)")

    return {
        "path": path.name,
        "max_frames": max_frames,
        "samples_per_frame": int(samples_per_frame),
        "inputs": {"latent": [1, VAE_DIM, max_frames]},
        "outputs": {"audio": [1, 1, int(total_samples)]},
    }


def export_vae_decoder_streaming(model, settings: ExportSettings) -> dict:
    """Export a stateful streaming VAE decoder with conv cache state buffers.

    Each of the 34 causal conv layers maintains a small context buffer as
    ct.StateType, allowing frame-by-frame decode with full temporal context.
    """
    from traceable_modules import TraceableStreamingVAEDecoder

    typer.echo("Exporting streaming VAE decoder (stateful, T=1)...")
    tokenizer = model.model.acoustic_tokenizer
    if tokenizer is None:
        from transformers import AutoModel
        tokenizer = AutoModel.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            subfolder="acoustic_tokenizer",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    wrapper = TraceableStreamingVAEDecoder(tokenizer.decoder).eval()

    # Trace with single input (caches are internal registered buffers)
    latent = torch.randn(1, VAE_DIM, 1)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (latent,), strict=False)

    # CoreML inputs + state types for each cache buffer
    inputs = [ct.TensorType(name="latent", shape=(1, VAE_DIM, 1), dtype=np.float32)]
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
        "inputs": {"latent": [1, VAE_DIM, 1]},
        "outputs": {"audio": [1, 1, 3200]},
        "cache_layers": [
            {"name": name, "channels": ch, "context_size": ctx}
            for name, _, ch, ctx, _ in wrapper.cache_layers
        ],
    }


def export_diffusion_head_b2(model, settings: ExportSettings) -> dict:
    """Export diffusion head with batch=2 for batched CFG (pos+neg in one call)."""
    from traceable_modules import TraceableDiffusionHead

    typer.echo("Exporting diffusion head (B=2)...")
    head = TraceableDiffusionHead(model.model.prediction_head).eval()

    noisy = torch.randn(2, VAE_DIM)
    timestep = torch.tensor([500.0, 500.0])
    condition = torch.randn(2, HIDDEN_SIZE)

    with torch.no_grad():
        traced = torch.jit.trace(head, (noisy, timestep, condition), strict=False)

    inputs = [
        ct.TensorType(name="noisy_latent", shape=(2, VAE_DIM), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(2,), dtype=np.float32),
        ct.TensorType(name="condition", shape=(2, HIDDEN_SIZE), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="predicted_noise", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_head_b2.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice diffusion head (B=2 batched CFG)")

    return {
        "path": path.name,
        "inputs": {"noisy_latent": [2, VAE_DIM], "timestep": [2], "condition": [2, HIDDEN_SIZE]},
        "outputs": {"predicted_noise": [2, VAE_DIM]},
    }


def export_vae_decoder_streaming_windowed(model, settings: ExportSettings, window_size: int = 6) -> dict:
    """Export streaming VAE decoder with T=window_size for windowed decode."""
    from traceable_modules import TraceableStreamingVAEDecoder

    typer.echo(f"Exporting streaming VAE decoder (stateful, T={window_size})...")
    tokenizer = model.model.acoustic_tokenizer
    if tokenizer is None:
        from transformers import AutoModel
        tokenizer = AutoModel.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            subfolder="acoustic_tokenizer",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    wrapper = TraceableStreamingVAEDecoder(tokenizer.decoder).eval()

    latent = torch.randn(1, VAE_DIM, window_size)
    with torch.no_grad():
        ref = wrapper(latent)
        frame_samples = ref.shape[-1]
    traced = torch.jit.trace(wrapper, (latent,), strict=False)

    inputs = [ct.TensorType(name="latent", shape=(1, VAE_DIM, window_size), dtype=np.float32)]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)]
    states = build_vae_state_specs(wrapper.cache_layers)

    coreml_model = coreml_convert(traced, inputs, outputs, settings,
                                   states=states,
                                   minimum_deployment_target=ct.target.iOS18)

    path = settings.output_dir / f"vae_decoder_streaming_w{window_size}.mlpackage"
    save_mlpackage(coreml_model, path,
                    f"VibeVoice streaming VAE decoder (stateful, T={window_size})")

    return {
        "path": path.name,
        "window_size": window_size,
        "frame_samples": int(frame_samples),
        "num_cache_layers": len(wrapper.cache_layers),
        "inputs": {"latent": [1, VAE_DIM, window_size]},
        "outputs": {"audio": [1, 1, int(frame_samples)]},
    }


def export_diffusion_loop(model, settings: ExportSettings,
                          num_steps: int = 5, cfg_scale: float = 1.5) -> dict:
    """Export fused DPM-Solver++ 2M diffusion loop with CFG for 0.5B.

    Replaces N*2 separate diffusion head calls with a single CoreML forward pass.
    Schedule constants are baked in; CFG scale is a runtime input.
    """
    from traceable_common import TraceableDiffusionLoopCFG

    typer.echo(f"Exporting fused diffusion loop (DPM-Solver++ 2M, {num_steps} steps, runtime CFG)...")
    loop = TraceableDiffusionLoopCFG(
        model.model.prediction_head,
        num_steps=num_steps,
    ).eval()

    noise = torch.randn(1, VAE_DIM)
    condition = torch.randn(1, HIDDEN_SIZE)
    neg_condition = torch.randn(1, HIDDEN_SIZE)
    cfg_scale_t = torch.tensor([cfg_scale])

    with torch.no_grad():
        traced = torch.jit.trace(loop, (noise, condition, neg_condition, cfg_scale_t), strict=False)

    inputs = [
        ct.TensorType(name="noise", shape=(1, VAE_DIM), dtype=np.float32),
        ct.TensorType(name="condition", shape=(1, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType(name="neg_condition", shape=(1, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType(name="cfg_scale", shape=(1,), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="latent", dtype=np.float32)]

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "diffusion_loop.mlpackage"
    save_mlpackage(coreml_model, path,
                   f"VibeVoice 0.5B fused diffusion loop (DPM++ 2M, {num_steps}s)")
    return {
        "path": path.name,
        "inputs": {
            "noise": [1, VAE_DIM],
            "condition": [1, HIDDEN_SIZE],
            "neg_condition": [1, HIDDEN_SIZE],
            "cfg_scale": [1],
        },
        "outputs": {"latent": [1, VAE_DIM]},
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

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "eos_classifier.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice EOS classifier (sigmoid probability)")

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

    coreml_model = coreml_convert(traced, inputs, outputs, settings)
    path = settings.output_dir / "acoustic_connector.mlpackage"
    save_mlpackage(coreml_model, path, "VibeVoice acoustic connector")

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
        Path(__file__).resolve().parent.parent / "build/vibevoice-realtime-0.5b",
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
    fused_diffusion: bool = typer.Option(False, "--fused-diffusion", help="Also export fused diffusion loop"),
    fused_diffusion_steps: int = typer.Option(5, "--fused-diffusion-steps", help="Steps for fused diffusion loop"),
    fused_diffusion_cfg: float = typer.Option(1.5, "--fused-diffusion-cfg", help="CFG scale for fused diffusion loop"),
) -> None:
    """Export VibeVoice-Realtime-0.5B components to CoreML."""
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=parse_compute_units(compute_units),
        compute_precision=parse_compute_precision(compute_precision),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_id)

    components = {}

    if not skip_diffusion:
        components["diffusion_head"] = export_diffusion_head(model, settings)
        components["diffusion_head_b2"] = export_diffusion_head_b2(model, settings)

    if fused_diffusion:
        components["diffusion_loop"] = export_diffusion_loop(
            model, settings,
            num_steps=fused_diffusion_steps, cfg_scale=fused_diffusion_cfg,
        )

    if not skip_vae:
        components["vae_decoder"] = export_vae_decoder(model, settings)
        components["vae_decoder_batch"] = export_vae_decoder_batch(model, settings)
        components["vae_decoder_streaming"] = export_vae_decoder_streaming(model, settings)
        components["vae_decoder_streaming_w6"] = export_vae_decoder_streaming_windowed(model, settings)

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
