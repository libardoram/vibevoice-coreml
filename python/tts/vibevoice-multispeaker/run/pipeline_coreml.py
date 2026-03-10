"""CoreML fp16/CPU+GPU backend for e2e TTS pipeline."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np

import pipeline_common as common
from diffusion import VAE_DIM
from rope import compute_rope_np


def run_coreml(
    input_ids: List[int],
    max_speech_tokens: int,
    seed: int,
    opt: common.OptConfig = None,
    voice_clone: Optional[common.VoiceCloneData] = None,
) -> Tuple[np.ndarray, common.PipelineMetrics]:
    """Full TTS pipeline using CoreML fp16/CPU+GPU."""
    if opt is None:
        opt = common.OptConfig()

    # Select model variant
    quant_suffix = "_int8" if opt.int8 else ""
    if opt.fused_lm_head:
        lm_name = f"lm_decoder_fused{quant_suffix}.mlpackage"
    else:
        lm_name = f"lm_decoder_stateful{quant_suffix}.mlpackage"

    fused = opt.fused_lm_head
    label = f"CoreML fp16/CPU+GPU ({opt.label})"
    metrics = common.PipelineMetrics(label)
    mem_before = common.get_peak_memory_mb()

    t0_total = time.perf_counter()

    # Load models
    t0 = time.perf_counter()

    # Pre-compute voice clone embeddings per speaker using CoreML encoder
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

    lm_path = common.BUILD_DIR / lm_name
    if not lm_path.exists():
        raise FileNotFoundError(f"{lm_path} not found. Run convert_all.py first.")
    lm_cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }
    lm_compute_units = lm_cu_map.get(opt.lm_compute, ct.ComputeUnit.ALL)
    cml_lm = ct.models.MLModel(str(lm_path), compute_units=lm_compute_units)
    cml_diff_loop = None
    if opt.fused_diffusion:
        diff_loop_path = common.BUILD_DIR / "diffusion_loop.mlpackage"
        if not diff_loop_path.exists():
            raise FileNotFoundError(f"{diff_loop_path} not found. Run convert_coreml.py --fused-diffusion first.")
        cml_diff_loop = ct.models.MLModel(str(diff_loop_path))
    cml_diff = None
    if cml_diff_loop is None:
        cml_diff = ct.models.MLModel(str(common.BUILD_DIR / "diffusion_head.mlpackage"))
    cml_vae = ct.models.MLModel(
        str(common.BUILD_DIR / "vae_decoder_streaming.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    vae_state = cml_vae.make_state()
    cml_conn = ct.models.MLModel(str(common.BUILD_DIR / "acoustic_connector.mlpackage"))
    cml_head = None if fused else ct.models.MLModel(str(common.BUILD_DIR / "lm_head.mlpackage"))
    embed_table = common.load_embeddings(common.BUILD_DIR / "embed_tokens.bin")

    # Streaming semantic encoder + connector (CoreML) for feedback loop
    cml_sem_enc = ct.models.MLModel(
        str(common.BUILD_DIR / "semantic_encoder_streaming.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    cml_sem_conn = ct.models.MLModel(str(common.BUILD_DIR / "semantic_connector.mlpackage"))
    sem_state = cml_sem_enc.make_state()

    # Warmup: trigger GPU compilation for all models
    warmup_state = cml_lm.make_state()
    dummy_h = np.zeros((1, 1, common.HIDDEN_SIZE), dtype=np.float32)
    dummy_cos = np.ones((1, 1, common.HEAD_DIM), dtype=np.float32)
    dummy_sin = np.zeros((1, 1, common.HEAD_DIM), dtype=np.float32)
    dummy_mask = np.zeros((1, 1, 1, 1), dtype=np.float32)
    cml_lm.predict({"hidden_states": dummy_h, "position_cos": dummy_cos, "position_sin": dummy_sin, "attention_mask": dummy_mask}, state=warmup_state)
    if cml_diff is not None:
        cml_diff.predict({"noisy_latent": np.zeros((1, VAE_DIM), dtype=np.float32), "timestep": np.array([0.0], dtype=np.float32), "condition": np.zeros((1, common.HIDDEN_SIZE), dtype=np.float32)})
    if cml_diff_loop is not None:
        cml_diff_loop.predict({"noise": np.zeros((1, VAE_DIM), dtype=np.float32), "condition": np.zeros((1, common.HIDDEN_SIZE), dtype=np.float32), "neg_condition": np.zeros((1, common.HIDDEN_SIZE), dtype=np.float32), "cfg_scale": np.array([opt.cfg_scale], dtype=np.float32)})
    warmup_vae_state = cml_vae.make_state()
    cml_vae.predict({"latent": np.zeros((1, VAE_DIM, 1), dtype=np.float32)}, state=warmup_vae_state)
    del warmup_vae_state
    cml_conn.predict({"speech_latent": np.zeros((1, 1, VAE_DIM), dtype=np.float32)})
    if cml_head:
        cml_head.predict({"hidden_state": dummy_h})
    cml_sem_enc.predict({"audio": np.zeros((1, 1, 3200), dtype=np.float32)}, state=sem_state)
    sem_state = cml_sem_enc.make_state()  # reset after warmup
    cml_sem_conn.predict({"semantic_features": np.zeros((1, 1, common.SEMANTIC_DIM), dtype=np.float32)})
    del warmup_state

    # Compute negative condition for CFG (speech_start token through LM)
    neg_state = cml_lm.make_state()
    neg_embed = embed_table[common.SPEECH_START_ID][None, None, :]
    neg_cos, neg_sin = compute_rope_np(0, common.HEAD_DIM)
    neg_hidden = cml_lm.predict({
        "hidden_states": neg_embed.astype(np.float32),
        "position_cos": neg_cos,
        "position_sin": neg_sin,
        "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }, state=neg_state)["output_hidden"]
    neg_condition_cml = neg_hidden[:, 0:1, :].reshape(1, common.HIDDEN_SIZE)
    del neg_state

    state = cml_lm.make_state()
    metrics.record("load", (time.perf_counter() - t0) * 1000)

    # Helper: get logits from hidden (fused vs separate)
    def get_hidden_and_token(lm_output):
        if fused:
            h = lm_output["output_hidden"]
            logits = lm_output["logits"]
        else:
            h = lm_output["output_hidden"]
            logits = cml_head.predict({"hidden_state": h})["logits"]
        return h, int(np.argmax(logits[0, 0]))

    # Prefill: batch all tokens at once
    t0 = time.perf_counter()
    seq_len = len(input_ids)
    embeds = embed_table[input_ids][None, :, :]

    # Inject voice cloning embeddings at speech_diffusion positions (per speaker)
    if voice_clone is not None:
        for spk in voice_clone.speakers:
            vc_embeds = spk._cached_embeds_coreml
            for i, pos in enumerate(spk.speech_embed_positions):
                if i < len(vc_embeds):
                    embeds[0, pos] = vc_embeds[i]

    from rope import ROPE_THETA
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, common.HEAD_DIM, 2, dtype=np.float32) / common.HEAD_DIM))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    cos_all = np.cos(freqs)[None, :, :].astype(np.float32)
    sin_all = np.sin(freqs)[None, :, :].astype(np.float32)

    mask = np.full((1, 1, seq_len, seq_len), -1e9, dtype=np.float32)
    mask = np.triu(mask, k=1)

    lm_out = cml_lm.predict({
        "hidden_states": embeds,
        "position_cos": cos_all,
        "position_sin": sin_all,
        "attention_mask": mask,
    }, state=state)
    if fused:
        hidden = lm_out["output_hidden"][:, -1:, :]
        logits = lm_out["logits"][:, -1:, :]
        next_token = int(np.argmax(logits[0, 0]))
    else:
        hidden = lm_out["output_hidden"][:, -1:, :]
        logits = cml_head.predict({"hidden_state": hidden})["logits"]
        next_token = int(np.argmax(logits[0, 0]))
    metrics.record("prefill", (time.perf_counter() - t0) * 1000)
    metrics.num_text_tokens = len(input_ids)

    # Diffusion function for sample_latent (non-fused path only)
    cml_guided_fn = None
    if cml_diff is not None:
        def cml_diffusion_fn(sample, timestep, condition):
            return cml_diff.predict({
                "noisy_latent": sample,
                "timestep": timestep,
                "condition": condition,
            })["predicted_noise"]
        cml_guided_fn = common.make_cfg_fn(cml_diffusion_fn, neg_condition_cml, opt.cfg_scale)

    # Autoregressive generation
    audio_chunks = []
    rng = np.random.RandomState(seed)
    position = len(input_ids)
    pool = ThreadPoolExecutor(max_workers=1) if opt.parallel else None
    pending_vae_future = None

    for step in range(max_speech_tokens * 3):
        if next_token == common.EOS_ID:
            break
        if metrics.num_speech_tokens >= max_speech_tokens:
            break

        if next_token == common.SPEECH_DIFFUSION_ID:
            metrics.num_speech_tokens += 1

            # Diffusion
            t0 = time.perf_counter()
            condition = hidden[:, 0:1, :].reshape(1, common.HIDDEN_SIZE)
            if cml_diff_loop is not None:
                # Match RNG pattern: consume one randint then seed inner RNG (same as non-fused path)
                inner_seed = rng.randint(0, 2**31)
                noise = np.random.RandomState(inner_seed).randn(1, VAE_DIM).astype(np.float32)
                sample = cml_diff_loop.predict({
                    "noise": noise,
                    "condition": condition,
                    "neg_condition": neg_condition_cml,
                    "cfg_scale": np.array([opt.cfg_scale], dtype=np.float32),
                })["latent"]
            else:
                sample = common._sample_latent(cml_guided_fn, condition, opt, seed=rng.randint(0, 2**31))
            metrics.record("diffusion", (time.perf_counter() - t0) * 1000)

            # Collect any pending VAE result from previous step
            if pending_vae_future is not None:
                audio_chunks.append(pending_vae_future.result())
                pending_vae_future = None

            # VAE decode (optionally async to overlap with next LM step)
            latent = sample / common.SPEECH_SCALING_FACTOR - common.SPEECH_BIAS_FACTOR
            if opt.parallel and pool is not None:
                def _vae_decode(lat):
                    t0v = time.perf_counter()
                    audio = cml_vae.predict({"latent": lat[:, :, None]}, state=vae_state)["audio"]
                    metrics.record("vae", (time.perf_counter() - t0v) * 1000)
                    return audio.squeeze()
                pending_vae_future = pool.submit(_vae_decode, latent.copy())
            else:
                t0 = time.perf_counter()
                audio = cml_vae.predict({"latent": latent[:, :, None]}, state=vae_state)["audio"]
                audio_chunks.append(audio.squeeze())
                metrics.record("vae", (time.perf_counter() - t0) * 1000)

            # Connectors: acoustic + semantic feedback
            t0 = time.perf_counter()
            acoustic_embed = cml_conn.predict({"speech_latent": sample[:, None, :]})["embedding"]
            if audio_chunks:
                # Streaming: feed only latest 3200-sample chunk (CoreML)
                chunk = audio_chunks[-1][:3200].astype(np.float32)
                audio_input = np.zeros((1, 1, 3200), dtype=np.float32)
                audio_input[0, 0, :len(chunk)] = chunk
                features = cml_sem_enc.predict({"audio": audio_input}, state=sem_state)["features"]
                feat = features.transpose(0, 2, 1)  # [1, 128, 1] -> [1, 1, 128]
                sem_embed = cml_sem_conn.predict({"semantic_features": feat})["embedding"]
                next_embed = acoustic_embed + sem_embed
            else:
                next_embed = acoustic_embed
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            if next_token == common.SPEECH_END_ID:
                # Reset streaming caches for new speaker segment
                vae_state = cml_vae.make_state()
                sem_state = cml_sem_enc.make_state()
            next_embed = embed_table[next_token][None, None, :]

        # LM step
        t0 = time.perf_counter()
        cos_np, sin_np = compute_rope_np(position, common.HEAD_DIM)
        mask = np.zeros((1, 1, 1, position + 1), dtype=np.float32)
        lm_out = cml_lm.predict({
            "hidden_states": next_embed.astype(np.float32),
            "position_cos": cos_np,
            "position_sin": sin_np,
            "attention_mask": mask,
        }, state=state)
        hidden, next_token = get_hidden_and_token(lm_out)
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000)
        position += 1

    # Collect final pending VAE
    if pending_vae_future is not None:
        audio_chunks.append(pending_vae_future.result())
    if pool is not None:
        pool.shutdown(wait=False)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    if audio_chunks:
        audio_out = np.concatenate(audio_chunks)
    else:
        audio_out = np.zeros(0, dtype=np.float32)
    metrics.audio_samples = len(audio_out)

    del cml_lm, cml_diff, cml_diff_loop, cml_vae, cml_conn, cml_head, vae_state, state
    del cml_sem_enc, cml_sem_conn, sem_state
    return audio_out, metrics
