"""CoreML backend for 0.5B streaming TTS pipeline."""

from __future__ import annotations

import math
import time
from typing import Tuple

import coremltools as ct
import numpy as np
import torch

import pipeline_common as common
from diffusion import VAE_DIM
from rope import compute_rope_np, compute_rope_np_multi


def _inject_kv_cache(cml_model, state, past_key_values, num_layers):
    """Inject pre-computed KV cache values into CoreML model state.

    Uses inject_mode=1 to write KV values directly into state buffers via
    the branchless selection: k = inj * k_inject + (1-inj) * k_computed.
    With zero hidden_states input, k_computed is finite (just projection bias),
    so 0 * finite = 0 and the injected values dominate.

    Returns:
        seq_len: length of injected sequence
    """
    seq_len = past_key_values[0][0].shape[2]

    inject_k = np.concatenate(
        [kv[0].float().numpy() for kv in past_key_values], axis=1
    ).astype(np.float32)
    inject_v = np.concatenate(
        [kv[1].float().numpy() for kv in past_key_values], axis=1
    ).astype(np.float32)

    cos_np, sin_np = compute_rope_np_multi(range(seq_len), common.HEAD_DIM)
    mask = np.full((1, 1, seq_len, seq_len), -1e9, dtype=np.float32)
    for i in range(seq_len):
        mask[0, 0, i, :i + 1] = 0.0

    dummy_hidden = np.zeros((1, seq_len, common.HIDDEN_SIZE), dtype=np.float32)

    cml_model.predict({
        "hidden_states": dummy_hidden,
        "position_cos": cos_np,
        "position_sin": sin_np,
        "attention_mask": mask,
        "inject_mode": np.array([1.0], dtype=np.float32),
        "inject_k": inject_k,
        "inject_v": inject_v,
    }, state=state)

    return seq_len


def run_coreml(text: str, voice_name: str, cfg_scale: float,
               diffusion_steps: int, seed: int,
               fused_diffusion: bool = False) -> Tuple[np.ndarray, common.Metrics]:
    """Run using CoreML models with KV cache injection from voice prompt."""
    metrics = common.Metrics()
    t0 = time.time()

    # Load CoreML models
    cml_base = ct.models.MLModel(str(common.BUILD_DIR / "base_lm_stateful.mlpackage"))
    cml_tts = ct.models.MLModel(str(common.BUILD_DIR / "tts_lm_stateful.mlpackage"))
    cml_vae = ct.models.MLModel(
        str(common.BUILD_DIR / "vae_decoder_streaming.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    cml_eos = ct.models.MLModel(str(common.BUILD_DIR / "eos_classifier.mlpackage"))
    cml_conn = ct.models.MLModel(str(common.BUILD_DIR / "acoustic_connector.mlpackage"))

    # Load fused diffusion loop if requested
    cml_diff_loop = None
    if fused_diffusion:
        diff_loop_path = common.BUILD_DIR / "diffusion_loop.mlpackage"
        if diff_loop_path.exists():
            cml_diff_loop = ct.models.MLModel(str(diff_loop_path))
            print("  Loaded fused diffusion loop")
        else:
            print(f"  Warning: {diff_loop_path} not found, falling back to per-step diffusion")

    # Load diffusion head (B=2 batched if available, else B=1)
    cml_diff = None
    cml_diff_b2 = None
    if cml_diff_loop is None:
        diff_b2_path = common.BUILD_DIR / "diffusion_head_b2.mlpackage"
        if diff_b2_path.exists():
            cml_diff_b2 = ct.models.MLModel(str(diff_b2_path))
            print("  Loaded B=2 batched diffusion head")
        else:
            cml_diff = ct.models.MLModel(str(common.BUILD_DIR / "diffusion_head.mlpackage"))
            print("  Loaded B=1 diffusion head")

    # Load constants
    embed_tokens = common.load_embeddings(common.BUILD_DIR / "embed_tokens.bin")
    tts_input_types = common.load_embeddings(common.BUILD_DIR / "tts_input_types.bin")
    speech_scaling = common.SPEECH_SCALING_FACTOR
    speech_bias = common.SPEECH_BIAS_FACTOR

    # Load voice prompt
    voice_path = common.load_voice_prompt(voice_name)
    prompt = torch.load(str(voice_path), map_location="cpu", weights_only=False)

    # Create states and inject KV caches
    base_state = cml_base.make_state()
    tts_state = cml_tts.make_state()
    neg_tts_state = cml_tts.make_state()
    vae_state = cml_vae.make_state()

    print("  Injecting KV caches from voice prompt...")
    t_inj = time.time()
    base_pos = _inject_kv_cache(cml_base, base_state,
                                prompt["lm"]["past_key_values"], common.BASE_LM_LAYERS)
    tts_pos = _inject_kv_cache(cml_tts, tts_state,
                               prompt["tts_lm"]["past_key_values"], common.TTS_LM_LAYERS)
    neg_tts_pos = _inject_kv_cache(cml_tts, neg_tts_state,
                                   prompt["neg_tts_lm"]["past_key_values"], common.TTS_LM_LAYERS)
    print(f"  Injected: base={base_pos}, tts={tts_pos}, neg_tts={neg_tts_pos} "
          f"({(time.time() - t_inj) * 1000:.0f}ms)")

    # Initial conditioning from prompt
    tts_last_hidden = prompt["tts_lm"]["last_hidden_state"][:, -1:, :].float().numpy()
    neg_tts_last_hidden = prompt["neg_tts_lm"]["last_hidden_state"][:, -1:, :].float().numpy()

    del prompt
    metrics.record("load", time.time() - t0)

    # Pre-allocate zero injection tensors for normal mode (inject_mode=0)
    base_total_kv = common.BASE_LM_LAYERS * common.NUM_KV_HEADS
    tts_total_kv = common.TTS_LM_LAYERS * common.NUM_KV_HEADS
    zero_inject_mode = np.zeros((1,), dtype=np.float32)

    # Tokenize text
    tts_text_ids = common.tokenize_text(text)
    text_window_idx = 0
    total_windows = math.ceil(len(tts_text_ids) / common.TTS_TEXT_WINDOW_SIZE)

    # Diffusion functions using CoreML
    def coreml_diffusion(sample, timestep, condition):
        out = cml_diff.predict({
            "noisy_latent": sample.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "condition": condition.astype(np.float32),
        })
        return out["predicted_noise"]

    def coreml_diffusion_b2(sample, timestep, condition):
        out = cml_diff_b2.predict({
            "noisy_latent": sample.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "condition": condition.astype(np.float32),
        })
        return out["predicted_noise"]

    # Helper to build predict dict with inject_mode=0
    def base_predict(hidden, cos, sin, mask):
        q = hidden.shape[1]
        return cml_base.predict({
            "hidden_states": hidden.astype(np.float32),
            "position_cos": cos,
            "position_sin": sin,
            "attention_mask": mask,
            "inject_mode": zero_inject_mode,
            "inject_k": np.zeros((1, base_total_kv, q, common.HEAD_DIM), dtype=np.float32),
            "inject_v": np.zeros((1, base_total_kv, q, common.HEAD_DIM), dtype=np.float32),
        }, state=base_state)

    def tts_predict(hidden, cos, sin, mask, state):
        q = hidden.shape[1]
        return cml_tts.predict({
            "hidden_states": hidden.astype(np.float32),
            "position_cos": cos,
            "position_sin": sin,
            "attention_mask": mask,
            "inject_mode": zero_inject_mode,
            "inject_k": np.zeros((1, tts_total_kv, q, common.HEAD_DIM), dtype=np.float32),
            "inject_v": np.zeros((1, tts_total_kv, q, common.HEAD_DIM), dtype=np.float32),
        }, state=state)

    # Generation loop — windowed VAE decode (T=6 per speech window)
    t0 = time.time()
    audio_chunks = []
    finished = False
    rng = np.random.RandomState(seed)

    while not finished:
        # Get current text window
        start = text_window_idx * common.TTS_TEXT_WINDOW_SIZE
        end = (text_window_idx + 1) * common.TTS_TEXT_WINDOW_SIZE
        cur_text_ids = tts_text_ids[start:end]
        text_window_idx += 1

        if len(cur_text_ids) > 0:
            # Embed text tokens
            text_embeds = embed_tokens[cur_text_ids]  # (Q, 896)
            text_embeds = text_embeds[None, :, :]  # (1, Q, 896)
            q = len(cur_text_ids)

            # Base LM forward for text window
            cos_np, sin_np = compute_rope_np_multi(range(base_pos, base_pos + q), common.HEAD_DIM)
            end_step = base_pos + q
            mask = np.full((1, 1, q, end_step), -1e9, dtype=np.float32)
            for i in range(q):
                mask[0, 0, i, :base_pos + i + 1] = 0.0

            t_lm = time.time()
            lm_out_dict = base_predict(text_embeds, cos_np, sin_np, mask)
            lm_last_hidden = lm_out_dict["output_hidden"]  # (1, Q, 896)
            base_pos += q
            metrics.record("base_lm", time.time() - t_lm)
            metrics.count("base_lm")

            # TTS LM forward for text window
            tts_embeds = lm_last_hidden + tts_input_types[1:2, :]  # text type
            cos_np, sin_np = compute_rope_np_multi(range(tts_pos, tts_pos + q), common.HEAD_DIM)
            end_step = tts_pos + q
            mask = np.full((1, 1, q, end_step), -1e9, dtype=np.float32)
            for i in range(q):
                mask[0, 0, i, :tts_pos + i + 1] = 0.0

            t_tts = time.time()
            tts_out_dict = tts_predict(tts_embeds, cos_np, sin_np, mask, tts_state)
            tts_last_hidden = tts_out_dict["output_hidden"][:, -1:, :]
            tts_pos += q
            metrics.record("tts_lm", time.time() - t_tts)
            metrics.count("tts_lm")

        # Speech frame generation loop
        for speech_idx in range(common.TTS_SPEECH_WINDOW_SIZE):
            pos_condition = tts_last_hidden[:, 0, :]  # (1, 896)
            neg_condition = neg_tts_last_hidden[:, 0, :]  # (1, 896)

            # Diffusion sampling with CFG
            t_diff = time.time()
            if cml_diff_loop is not None:
                # Fused diffusion loop: single CoreML call for all DPM steps
                inner_seed = rng.randint(0, 2**31) if seed is not None else None
                if inner_seed is not None:
                    noise = np.random.RandomState(inner_seed).randn(1, VAE_DIM).astype(np.float32)
                else:
                    noise = np.random.randn(1, VAE_DIM).astype(np.float32)
                speech_latent = cml_diff_loop.predict({
                    "noise": noise,
                    "condition": pos_condition,
                    "neg_condition": neg_condition,
                    "cfg_scale": np.array([cfg_scale], dtype=np.float32),
                })["latent"]  # (1, 64)
            elif cml_diff_b2 is not None:
                # Batched CFG: single B=2 call per diffusion step
                guided_fn = common.make_batched_cfg_fn(coreml_diffusion_b2, neg_condition, cfg_scale)
                speech_latent = common.dpm_solver_2m_sample(
                    guided_fn, pos_condition, diffusion_steps,
                    seed=rng.randint(0, 2**31) if seed is not None else None,
                )  # (1, 64)
            else:
                # Original: two B=1 calls per diffusion step
                guided_fn = common.make_cfg_fn(coreml_diffusion, neg_condition, cfg_scale)
                speech_latent = common.dpm_solver_2m_sample(
                    guided_fn, pos_condition, diffusion_steps,
                    seed=rng.randint(0, 2**31) if seed is not None else None,
                )  # (1, 64)
            metrics.record("diffusion", time.time() - t_diff)
            metrics.count("diffusion")

            # Scale latent and decode via streaming VAE (frame-by-frame with state)
            scaled_latent = speech_latent / speech_scaling - speech_bias
            latent_3d = scaled_latent.T[None, :, :]  # (1, 64) -> (1, 64, 1)
            t_vae = time.time()
            audio_frame = cml_vae.predict(
                {"latent": latent_3d.astype(np.float32)},
                state=vae_state,
            )["audio"]
            audio_chunks.append(audio_frame.flatten())
            metrics.record("vae", time.time() - t_vae)
            metrics.count("vae")

            # Acoustic connector for feedback
            t_conn = time.time()
            speech_embed = cml_conn.predict({
                "speech_latent": speech_latent[:, None, :].astype(np.float32),
            })["embedding"]  # (1, 1, 896)
            metrics.record("connector", time.time() - t_conn)

            # TTS LM forward with speech embedding (type=speech=0)
            tts_embeds = speech_embed + tts_input_types[0:1, :]
            cos_np, sin_np = compute_rope_np(tts_pos, common.HEAD_DIM)
            end_step = tts_pos + 1
            mask = np.zeros((1, 1, 1, end_step), dtype=np.float32)

            t_tts = time.time()
            tts_out_dict = tts_predict(tts_embeds, cos_np, sin_np, mask, tts_state)
            tts_last_hidden = tts_out_dict["output_hidden"][:, -1:, :]
            tts_pos += 1
            metrics.record("tts_lm", time.time() - t_tts)
            metrics.count("tts_lm")

            # Negative TTS LM forward
            neg_embeds = speech_embed + tts_input_types[0:1, :]
            cos_np, sin_np = compute_rope_np(neg_tts_pos, common.HEAD_DIM)
            end_step = neg_tts_pos + 1
            mask = np.zeros((1, 1, 1, end_step), dtype=np.float32)

            neg_tts_out_dict = tts_predict(neg_embeds, cos_np, sin_np, mask, neg_tts_state)
            neg_tts_last_hidden = neg_tts_out_dict["output_hidden"][:, -1:, :]
            neg_tts_pos += 1

            # EOS check
            t_eos = time.time()
            eos_prob = cml_eos.predict({
                "hidden_state": tts_last_hidden[:, 0, :].astype(np.float32),
            })["eos_probability"]
            metrics.record("eos", time.time() - t_eos)

            if float(eos_prob.flatten()[0]) > 0.5:
                finished = True
                break

        # Safety: limit total speech frames
        max_speech_frames = max(len(tts_text_ids) * 5, 30)
        if len(audio_chunks) >= max_speech_frames:
            break

        # Check if we've exhausted all text windows
        if text_window_idx >= total_windows and not finished:
            if text_window_idx > total_windows + 10:
                break

    gen_time = time.time() - t0
    metrics.record("generate", gen_time)

    # Concatenate audio frames from streaming VAE decode
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
    else:
        audio = np.zeros(common.SAMPLE_RATE, dtype=np.float32)

    return audio, metrics
