"""CoreML backend for ASR pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np

import pipeline_common as common


def run_coreml(
    audio_path: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 4096,
    int8: bool = False,
    fused_lm_head: bool = False,
) -> tuple[str, common.PipelineMetrics]:
    """Full ASR pipeline using CoreML models.

    Returns (transcription_text, metrics).
    """
    label_parts = []
    if int8:
        label_parts.append("int8")
    if fused_lm_head:
        label_parts.append("fused")
    label = "+".join(label_parts) if label_parts else "fp16"
    metrics = common.PipelineMetrics(f"CoreML ({label})")
    mem_before = common.get_peak_memory_mb()
    t0_total = time.perf_counter()

    build_dir = common.BUILD_DIR

    # Load CoreML models
    t0 = time.perf_counter()
    fused_enc_path = build_dir / "fused_encoder.mlpackage"
    use_fused_enc = fused_enc_path.exists()
    if use_fused_enc:
        fused_enc = ct.models.MLModel(str(fused_enc_path),
                                       compute_units=ct.ComputeUnit.CPU_AND_GPU)
        acoustic_enc = semantic_enc = None
    else:
        fused_enc = None
        acoustic_enc = ct.models.MLModel(
            str(build_dir / "acoustic_encoder.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        semantic_enc = ct.models.MLModel(
            str(build_dir / "semantic_encoder.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
    fused_proj_path = build_dir / "fused_projector.mlpackage"
    use_fused_proj = fused_proj_path.exists()
    if use_fused_proj:
        fused_proj = ct.models.MLModel(str(fused_proj_path))
        acoustic_proj = semantic_proj = None
    else:
        fused_proj = None
        acoustic_proj = ct.models.MLModel(
            str(build_dir / "acoustic_projector.mlpackage"),
        )
        semantic_proj = ct.models.MLModel(
            str(build_dir / "semantic_projector.mlpackage"),
        )

    # LM decoder
    if fused_lm_head:
        lm_name = "lm_decoder_fused_int8" if int8 else "lm_decoder_fused"
    elif int8:
        lm_name = "lm_decoder_stateful_int8"
    else:
        lm_name = "lm_decoder_stateful"
    lm_decoder = ct.models.MLModel(
        str(build_dir / f"{lm_name}.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    lm_state = lm_decoder.make_state()

    # LM head (only if not fused)
    lm_head_model = None
    if not fused_lm_head:
        lm_head_model = ct.models.MLModel(
            str(build_dir / "lm_head.mlpackage"),
        )

    # Embeddings
    embed_table = common.load_embeddings(build_dir / "embed_tokens.bin")

    # Pre-warm tokenizer + prompt constants (avoids counting it in prompt_build)
    common._ensure_prompt_constants()

    metrics.record("load", (time.perf_counter() - t0) * 1000)

    # Load and preprocess audio
    t0 = time.perf_counter()
    wav = common.load_audio(audio_path)
    duration_secs = len(wav) / common.SAMPLE_RATE
    num_audio_tokens = math.ceil(len(wav) / common.HOP_LENGTH)
    metrics.num_audio_tokens = num_audio_tokens
    metrics.record("audio_load", (time.perf_counter() - t0) * 1000)

    # Encode audio (chunked, 60s per chunk)
    t0 = time.perf_counter()
    num_chunks = math.ceil(len(wav) / common.CHUNK_SAMPLES)
    acoustic_features_list = []
    semantic_features_list = []

    for c in range(num_chunks):
        start = c * common.CHUNK_SAMPLES
        end = min(start + common.CHUNK_SAMPLES, len(wav))
        chunk = np.zeros(common.CHUNK_SAMPLES, dtype=np.float32)
        chunk[:end - start] = wav[start:end]
        audio_input = chunk.reshape(1, 1, common.CHUNK_SAMPLES)

        if use_fused_enc:
            out = fused_enc.predict({"audio": audio_input})
            ac_out = out["acoustic_features"]
            sem_out = out["semantic_features"]
        else:
            ac_out = acoustic_enc.predict({"audio": audio_input})["features"]
            sem_out = semantic_enc.predict({"audio": audio_input})["features"]

        # Trim to actual tokens for this chunk
        chunk_tokens = min(common.CHUNK_TOKENS, math.ceil((end - start) / common.HOP_LENGTH))
        acoustic_features_list.append(ac_out[:, :chunk_tokens, :])
        semantic_features_list.append(sem_out[:, :chunk_tokens, :])

    acoustic_features = np.concatenate(acoustic_features_list, axis=1)  # (1, T, 64)
    semantic_features = np.concatenate(semantic_features_list, axis=1)  # (1, T, 128)

    # VAE noise injection (matches HF inference: per-batch scale * per-element noise)
    rng = np.random.RandomState(42)
    noise_std = common.VAE_STD * rng.randn(1).astype(np.float32)  # per-batch scale
    acoustic_features = acoustic_features + noise_std[:, None, None] * rng.randn(*acoustic_features.shape).astype(np.float32)

    # Project to LM space
    if use_fused_proj:
        audio_embeddings = fused_proj.predict({
            "acoustic_features": acoustic_features.astype(np.float32),
            "semantic_features": semantic_features.astype(np.float32),
        })["embedding"].astype(np.float32)  # (1, T, 3584)
    else:
        ac_embed = acoustic_proj.predict({"features": acoustic_features.astype(np.float32)})["embedding"]
        sem_embed = semantic_proj.predict({"features": semantic_features.astype(np.float32)})["embedding"]
        audio_embeddings = (ac_embed + sem_embed).astype(np.float32)  # (1, T, 3584)
    metrics.record("encode", (time.perf_counter() - t0) * 1000)

    # Build prompt embeddings
    t0 = time.perf_counter()
    input_ids = common.build_prompt_ids(num_audio_tokens, duration_secs, prompt=prompt)
    metrics.num_prompt_tokens = len(input_ids)

    # Vectorized: look up all text embeddings at once, then splice in audio embeddings
    ids_array = np.array(input_ids, dtype=np.int32)

    # Gather all token embeddings (audio placeholder positions get overwritten)
    all_embeds = embed_table[ids_array]  # (S, 3584)

    # Overwrite audio token positions with projected audio embeddings
    audio_positions = np.where(ids_array == common.AUDIO_TOKEN_ID)[0]
    num_audio = min(len(audio_positions), audio_embeddings.shape[1])
    all_embeds[audio_positions[:num_audio]] = audio_embeddings[0, :num_audio, :]

    all_embeds = all_embeds.reshape(1, len(input_ids), common.HIDDEN_SIZE).astype(np.float32)
    metrics.record("prompt_build", (time.perf_counter() - t0) * 1000)

    # Batch prefill: feed all prompt embeddings in one call
    t0 = time.perf_counter()
    seq_len = all_embeds.shape[1]
    cos, sin = common.compute_rope_batch_np(0, seq_len, common.HEAD_DIM)
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    # Causal mask: position i can attend to positions 0..i
    for i in range(seq_len):
        mask[0, 0, i, i + 1:] = -1e9

    lm_input = {
        "hidden_states": all_embeds,
        "position_cos": cos,
        "position_sin": sin,
        "attention_mask": mask,
    }
    out = lm_decoder.predict(lm_input, state=lm_state)

    if fused_lm_head:
        last_logits = out["logits"][:, -1:, :]
    else:
        last_hidden = out["output_hidden"][:, -1:, :]
        last_logits = lm_head_model.predict(
            {"hidden_state": last_hidden.astype(np.float32)}
        )["logits"]

    metrics.record("prefill", (time.perf_counter() - t0) * 1000)

    # Precompute RoPE table for all decode positions
    max_position = seq_len + max_new_tokens
    rope_cos_table, rope_sin_table = common.compute_rope_batch_np(0, max_position, common.HEAD_DIM)
    # rope_cos_table: (1, max_position, HEAD_DIM), slice per step

    # Pre-allocate mask buffer at max size, slice per step
    max_mask_len = seq_len + max_new_tokens + 1
    mask_buffer = np.zeros((1, 1, 1, max_mask_len), dtype=np.float32)

    # Autoregressive generation
    next_token = int(np.argmax(last_logits[0, 0]))
    generated_ids = []
    position = seq_len

    for step in range(max_new_tokens):
        if next_token == common.EOS_ID:
            break

        generated_ids.append(next_token)

        t0 = time.perf_counter()
        hidden = embed_table[next_token:next_token + 1].reshape(1, 1, common.HIDDEN_SIZE).astype(np.float32)
        cos = rope_cos_table[:, position:position + 1, :]
        sin = rope_sin_table[:, position:position + 1, :]
        mask = mask_buffer[:, :, :, :position + 1]

        out = lm_decoder.predict({
            "hidden_states": hidden,
            "position_cos": cos,
            "position_sin": sin,
            "attention_mask": mask,
        }, state=lm_state)

        if fused_lm_head:
            logits = out["logits"]
        else:
            logits = lm_head_model.predict(
                {"hidden_state": out["output_hidden"].astype(np.float32)}
            )["logits"]

        next_token = int(np.argmax(logits[0, 0]))
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000)
        position += 1

    metrics.num_generated_tokens = len(generated_ids)

    # Decode tokens to text (reuse cached tokenizer from prompt build)
    tokenizer = common._get_tokenizer()
    transcription = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    return transcription, metrics
