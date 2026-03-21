"""MLX backend for e2e TTS pipeline, using vibevoice-mlx library."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np

import mlx.core as mx

import pipeline_common as common
from vibevoice_mlx import load_model, generate, GenerationOptions


def run_mlx(
    input_ids: List[int],
    max_speech_tokens: int,
    seed: int,
    opt: common.OptConfig = None,
    voice_clone: Optional[common.VoiceCloneData] = None,
) -> Tuple[np.ndarray, common.PipelineMetrics]:
    """Full TTS pipeline using vibevoice-mlx library with CoreML semantic encoder."""
    if opt is None:
        opt = common.OptConfig()
    quant_bits = 8 if opt.int8 else None
    quant_label = "int8" if opt.int8 else "fp16"
    metrics = common.PipelineMetrics(f"MLX {quant_label}/GPU ({opt.solver}-{opt.diffusion_steps}s)")
    mem_before = common.get_peak_memory_mb()

    t0_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    mlx_model_id = common.MODEL_CONFIGS[common.MODEL_ID].get("mlx_model_id", common.MODEL_ID)
    model, config = load_model(mlx_model_id, quantize_bits=quant_bits)
    metrics.record("load", (time.perf_counter() - t0) * 1000)

    # CoreML streaming semantic encoder + connector
    cml_sem_enc = ct.models.MLModel(
        str(common.BUILD_DIR / "semantic_encoder_streaming.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    cml_sem_conn = ct.models.MLModel(
        str(common.BUILD_DIR / "semantic_connector.mlpackage"),
    )
    sem_state = cml_sem_enc.make_state()

    def semantic_fn(audio_chunk: np.ndarray) -> np.ndarray:
        """Encode audio chunk via CoreML semantic encoder + connector."""
        audio_input = np.zeros((1, 1, 3200), dtype=np.float32)
        audio_input[0, 0, :len(audio_chunk)] = audio_chunk
        features = cml_sem_enc.predict(
            {"audio": audio_input}, state=sem_state
        )["features"]
        feat = features.transpose(0, 2, 1)
        return cml_sem_conn.predict(
            {"semantic_features": feat}
        )["embedding"]  # (1, 1, hidden_size)

    def semantic_reset():
        nonlocal sem_state
        sem_state = cml_sem_enc.make_state()

    # Build voice_embeds dict from VoiceCloneData
    voice_embeds = None
    if voice_clone is not None:
        needs_encode = any(spk._cached_embeds_coreml is None for spk in voice_clone.speakers)
        if needs_encode:
            _vae_enc = ct.models.MLModel(str(common.BUILD_DIR / "vae_encoder.mlpackage"))
            _ac_conn = ct.models.MLModel(str(common.BUILD_DIR / "acoustic_connector.mlpackage"))
            for spk in voice_clone.speakers:
                if spk._cached_embeds_coreml is None:
                    spk._cached_embeds_coreml = common.encode_voice_reference_coreml(
                        spk.ref_audio_np, spk.num_vae_tokens, _vae_enc, _ac_conn)
            del _vae_enc, _ac_conn

        voice_embeds = {}
        for spk in voice_clone.speakers:
            spk_embeds_mx = mx.array(spk._cached_embeds_coreml).astype(mx.float16)
            for i, pos in enumerate(spk.speech_embed_positions):
                if i < spk_embeds_mx.shape[0]:
                    voice_embeds[pos] = spk_embeds_mx[i:i + 1]

    # Generate
    gen_opts = GenerationOptions(
        solver=opt.solver,
        diffusion_steps=opt.diffusion_steps,
        cfg_scale=opt.cfg_scale,
        max_speech_tokens=max_speech_tokens,
        seed=seed,
    )

    audio_out, gen_metrics = generate(
        model, input_ids, opts=gen_opts,
        semantic_encoder_fn=semantic_fn,
        semantic_reset_fn=semantic_reset,
        voice_embeds=voice_embeds,
    )

    # Map GenerationMetrics to PipelineMetrics
    for comp, timings in gen_metrics.timings.items():
        for ms in timings:
            metrics.record(comp, ms)
    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.num_speech_tokens = gen_metrics.num_speech_tokens
    metrics.num_text_tokens = gen_metrics.num_text_tokens
    metrics.audio_samples = gen_metrics.audio_samples
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    return audio_out, metrics
