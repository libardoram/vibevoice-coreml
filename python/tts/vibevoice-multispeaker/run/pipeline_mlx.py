"""MLX fp16/GPU backend for e2e TTS pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np

import mlx.core as mx
import mlx.nn as nn_mlx

import pipeline_common as common

from bench_mlx_common import (
    AcousticConnector as MlxAcousticConnector,
    DiffusionHead as MlxDiffusionHead,
    LMHead as MlxLMHead,
    Qwen2Decoder as MlxQwen2Decoder,
    Qwen2Layer as MlxQwen2Layer,
    VAEDecoder as MlxVAEDecoder,
    make_rope as mlx_make_rope,
    rms_norm as mlx_rms_norm,
    rotate_half as mlx_rotate_half,
    qmm,
    to_mx,
)
from diffusion import DDPM_STEPS, VAE_DIM


# Precompute DPM-Solver++ schedule constants (float64 for precision)
_AC64 = np.cos((np.arange(DDPM_STEPS + 1, dtype=np.float64) / DDPM_STEPS + 0.008) / 1.008 * np.pi / 2) ** 2
_AC64 = (_AC64 / _AC64[0])[:DDPM_STEPS]
_ALPHA_NP = np.sqrt(_AC64)
_SIGMA_NP = np.sqrt(1.0 - _AC64)
_LAMBDA_NP = np.log(_ALPHA_NP / np.maximum(_SIGMA_NP, 1e-10))


def dpm_solver_2m_sample_mlx(
    diff_head: MlxDiffusionHead,
    condition_mx: mx.array,
    neg_condition_mx: mx.array,
    cfg_scale: float,
    num_steps: int = 10,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """DPM-Solver++ 2M entirely in MLX with batched CFG (B=2).

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)

    key = mx.random.key(seed)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=key).astype(mx.float32)

    # Precompute batched conditions: [condition, neg_condition] along B=0
    # condition_mx: (1, H), neg_condition_mx: (1, H)
    batched_cond = mx.concatenate([condition_mx.astype(dtype), neg_condition_mx.astype(dtype)], axis=0)  # (2, H)

    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        # Batched CFG: concat [sample, sample] for B=2
        batched_sample = mx.concatenate([sample, sample], axis=0).astype(dtype)  # (2, VAE_DIM)
        ts_mx = mx.array([float(s)]).astype(dtype)  # (1,) — broadcasted

        v_batched = diff_head(batched_sample, ts_mx, batched_cond)  # (2, VAE_DIM)
        mx.eval(v_batched)

        v_cond = v_batched[0:1].astype(mx.float32)    # (1, VAE_DIM)
        v_uncond = v_batched[1:2].astype(mx.float32)   # (1, VAE_DIM)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        alpha_s = float(_ALPHA_NP[s])
        sigma_s = float(_SIGMA_NP[s])
        x0 = alpha_s * sample - sigma_s * v
        x0_list.append(x0)

        lam_s = float(_LAMBDA_NP[s])
        lam_t = float(_LAMBDA_NP[max(t, 0)])
        h = lam_t - lam_s

        is_last = (i == num_steps - 1)
        is_second_last = (i == num_steps - 2)
        lower_order_final = is_last and num_steps < 15
        lower_order_second = is_second_last and num_steps < 15
        use_first_order = len(x0_list) < 2 or lower_order_final or lower_order_second

        if use_first_order:
            D = x0_list[-1]
        else:
            s_prev = int(t_schedule[i - 1])
            lam_s_prev = float(_LAMBDA_NP[s_prev])
            h_prev = lam_s - lam_s_prev
            r = h_prev / h
            D0 = x0_list[-1]
            D1 = (1.0 / r) * (x0_list[-1] - x0_list[-2])
            D = D0 + 0.5 * D1

        sigma_t = float(_SIGMA_NP[t])
        alpha_t = float(_ALPHA_NP[t])
        sample = (sigma_t / sigma_s) * sample - alpha_t * float(np.expm1(-h)) * D

    return sample


def run_mlx(
    input_ids: List[int],
    max_speech_tokens: int,
    seed: int,
    weights: dict,
    opt: common.OptConfig = None,
    voice_clone: Optional[common.VoiceCloneData] = None,
) -> Tuple[np.ndarray, common.PipelineMetrics]:
    """Full TTS pipeline using MLX fp16/GPU with manual KV cache management."""
    if opt is None:
        opt = common.OptConfig()
    dtype = mx.float16
    use_int8 = opt.int8
    quant_label = "int8" if use_int8 else "fp16"
    metrics = common.PipelineMetrics(f"MLX {quant_label}/GPU ({opt.solver}-{opt.diffusion_steps}s)")
    mem_before = common.get_peak_memory_mb()

    t0_total = time.perf_counter()

    # Load components
    t0 = time.perf_counter()
    diff_head = MlxDiffusionHead(weights, dtype)
    vae_decoder = MlxVAEDecoder(weights, dtype)
    ac_conn = MlxAcousticConnector(weights, dtype)
    lm_head_key = "lm_head.weight" if "lm_head.weight" in weights else "model.language_model.embed_tokens.weight"
    lm_head = MlxLMHead(weights, key=lm_head_key, dtype=dtype, quantize=use_int8)

    # Streaming semantic encoder + connector (CoreML)
    cml_sem_enc = ct.models.MLModel(
        str(common.BUILD_DIR / "semantic_encoder_streaming.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    cml_sem_conn = ct.models.MLModel(
        str(common.BUILD_DIR / "semantic_connector.mlpackage"),
    )
    sem_state = cml_sem_enc.make_state()

    embed_table_np = weights["model.language_model.embed_tokens.weight"].float().numpy()
    embed_table_mx = mx.array(embed_table_np).astype(dtype)

    # Build LM layers (manual KV cache)
    lm_layers = []
    for i in range(common.NUM_LAYERS):
        lm_layers.append(MlxQwen2Layer(weights, f"model.language_model.layers.{i}.", dtype, quantize=use_int8))
    lm_norm_w = to_mx(weights["model.language_model.norm.weight"], dtype)

    metrics.record("load", (time.perf_counter() - t0) * 1000)

    def lm_forward_with_cache(h_mx, cos_mx, sin_mx, k_cache, v_cache, pos):
        """Single-token LM forward with manual KV cache."""
        B, Q, H = h_mx.shape
        for li, layer in enumerate(lm_layers):
            residual = h_mx
            h = mlx_rms_norm(h_mx, common.RMS_NORM_EPS, layer.input_ln_w)

            q = (qmm(h, layer.q_proj) + layer.q_bias).reshape(B, Q, common.NUM_Q_HEADS, common.HEAD_DIM).transpose(0, 2, 1, 3)
            k = (qmm(h, layer.k_proj) + layer.k_bias).reshape(B, Q, common.NUM_KV_HEADS, common.HEAD_DIM).transpose(0, 2, 1, 3)
            v = (qmm(h, layer.v_proj) + layer.v_bias).reshape(B, Q, common.NUM_KV_HEADS, common.HEAD_DIM).transpose(0, 2, 1, 3)

            from rope import ROPE_THETA
            q = q * cos_mx + mlx_rotate_half(q) * sin_mx
            k = k * cos_mx + mlx_rotate_half(k) * sin_mx

            # Update cache
            if k_cache[li] is None:
                k_cache[li] = k
                v_cache[li] = v
            else:
                k_cache[li] = mx.concatenate([k_cache[li], k], axis=2)
                v_cache[li] = mx.concatenate([v_cache[li], v], axis=2)

            k_full = mx.repeat(k_cache[li], common.GQA_REPEAT, axis=1)
            v_full = mx.repeat(v_cache[li], common.GQA_REPEAT, axis=1)

            # Attention in float32 to prevent fp16 overflow in QK^T
            scale = 1.0 / math.sqrt(common.HEAD_DIM)
            attn = (q.astype(mx.float32) @ k_full.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
            attn = mx.softmax(attn, axis=-1)
            out = (attn @ v_full.astype(mx.float32)).astype(dtype).transpose(0, 2, 1, 3).reshape(B, Q, H)
            h_mx = residual + qmm(out, layer.o_proj)

            residual = h_mx
            h = mlx_rms_norm(h_mx, common.RMS_NORM_EPS, layer.post_ln_w)
            gate = nn_mlx.silu(qmm(h, layer.gate_proj))
            up = qmm(h, layer.up_proj)
            h_mx = residual + qmm(gate * up, layer.down_proj)

        h_mx = mlx_rms_norm(h_mx, common.RMS_NORM_EPS, lm_norm_w)
        return h_mx

    # Compute negative condition for CFG (speech_start token through LM, no cache)
    neg_k = [None] * common.NUM_LAYERS
    neg_v = [None] * common.NUM_LAYERS
    neg_embed_mx = embed_table_mx[common.SPEECH_START_ID].reshape(1, 1, common.HIDDEN_SIZE)
    from rope import ROPE_THETA
    neg_cos_mx, neg_sin_mx = mlx_make_rope(mx.array(0.0), common.HEAD_DIM, ROPE_THETA, dtype)
    neg_hidden_mx = lm_forward_with_cache(neg_embed_mx, neg_cos_mx, neg_sin_mx, neg_k, neg_v, 0)
    mx.eval(neg_hidden_mx)
    neg_condition_mx = neg_hidden_mx[:, 0:1, :].reshape(1, common.HIDDEN_SIZE)
    del neg_k, neg_v

    # Prefill
    t0 = time.perf_counter()
    k_cache = [None] * common.NUM_LAYERS
    v_cache = [None] * common.NUM_LAYERS

    # Pre-compute voice clone embeddings per speaker (CoreML VAE encoder)
    vc_pos_to_embed = {}  # position -> MLX embed array
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
        for spk in voice_clone.speakers:
            spk_embeds_mx = mx.array(spk._cached_embeds_coreml).astype(dtype)
            for i, pos in enumerate(spk.speech_embed_positions):
                if i < spk_embeds_mx.shape[0]:
                    vc_pos_to_embed[pos] = spk_embeds_mx[i:i+1]

    for pos, tok_id in enumerate(input_ids):
        if pos in vc_pos_to_embed:
            embed = vc_pos_to_embed[pos].reshape(1, 1, common.HIDDEN_SIZE)
        else:
            embed = embed_table_mx[tok_id].reshape(1, 1, common.HIDDEN_SIZE)
        cos_mx, sin_mx = mlx_make_rope(mx.array(float(pos)), common.HEAD_DIM, ROPE_THETA, dtype)
        hidden_mx = lm_forward_with_cache(embed, cos_mx, sin_mx, k_cache, v_cache, pos)
        mx.eval(hidden_mx)

    metrics.record("prefill", (time.perf_counter() - t0) * 1000)
    metrics.num_text_tokens = len(input_ids)

    # First token
    logits_mx = lm_head(hidden_mx)
    mx.eval(logits_mx)
    next_token = int(mx.argmax(logits_mx[0, 0]).item())

    # Autoregressive generation
    audio_chunks = []
    rng = np.random.RandomState(seed)
    position = len(input_ids)

    for step in range(max_speech_tokens * 3):
        if next_token == common.EOS_ID:
            break
        if metrics.num_speech_tokens >= max_speech_tokens:
            break

        if next_token == common.SPEECH_DIFFUSION_ID:
            metrics.num_speech_tokens += 1

            # Diffusion (pure MLX, batched CFG)
            t0 = time.perf_counter()
            condition_mx = hidden_mx[:, 0:1, :].reshape(1, common.HIDDEN_SIZE)
            sample_mx = dpm_solver_2m_sample_mlx(
                diff_head, condition_mx, neg_condition_mx, opt.cfg_scale,
                num_steps=opt.diffusion_steps,
                seed=rng.randint(0, 2**31),
                dtype=dtype,
            )
            metrics.record("diffusion", (time.perf_counter() - t0) * 1000)

            # VAE decode
            t0 = time.perf_counter()
            latent_mx = (sample_mx / common.SPEECH_SCALING_FACTOR - common.SPEECH_BIAS_FACTOR)
            latent_mx = latent_mx[:, :, None].astype(dtype)
            audio_mx = vae_decoder(latent_mx)
            mx.eval(audio_mx)
            audio_chunks.append(np.array(audio_mx).squeeze().astype(np.float32))
            metrics.record("vae", (time.perf_counter() - t0) * 1000)

            # Connectors: acoustic + streaming semantic feedback (CoreML)
            t0 = time.perf_counter()
            acoustic_embed_mx = ac_conn(sample_mx[:, None, :].astype(dtype))
            if audio_chunks:
                # Streaming semantic encoder via CoreML
                chunk = audio_chunks[-1][:3200].astype(np.float32)
                audio_input = np.zeros((1, 1, 3200), dtype=np.float32)
                audio_input[0, 0, :len(chunk)] = chunk
                features = cml_sem_enc.predict(
                    {"audio": audio_input}, state=sem_state
                )["features"]
                # features shape: (1, sem_dim, 1) -> (1, 1, sem_dim)
                feat = features.transpose(0, 2, 1)
                sem_embed = cml_sem_conn.predict(
                    {"semantic_features": feat}
                )["embedding"]  # (1, 1, hidden_size)
                next_embed_mx = acoustic_embed_mx + mx.array(sem_embed).astype(dtype)
            else:
                next_embed_mx = acoustic_embed_mx
            mx.eval(next_embed_mx)
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            next_embed_mx = embed_table_mx[next_token].reshape(1, 1, common.HIDDEN_SIZE)

        # LM step
        t0 = time.perf_counter()
        cos_mx, sin_mx = mlx_make_rope(mx.array(float(position)), common.HEAD_DIM, ROPE_THETA, dtype)
        hidden_mx = lm_forward_with_cache(next_embed_mx, cos_mx, sin_mx, k_cache, v_cache, position)
        logits_mx = lm_head(hidden_mx)
        mx.eval(logits_mx)
        next_token = int(mx.argmax(logits_mx[0, 0]).item())
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000)
        position += 1

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    if audio_chunks:
        audio_out = np.concatenate(audio_chunks)
    else:
        audio_out = np.zeros(0, dtype=np.float32)
    metrics.audio_samples = len(audio_out)

    return audio_out, metrics
