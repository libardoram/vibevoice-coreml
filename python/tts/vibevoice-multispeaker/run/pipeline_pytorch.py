"""PyTorch fp32/CPU backend for e2e TTS pipeline."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert"))

import pipeline_common as common


def run_pytorch(
    input_ids: List[int],
    max_speech_tokens: int,
    seed: int,
    opt: common.OptConfig = None,
    voice_clone: Optional[common.VoiceCloneData] = None,
) -> Tuple[np.ndarray, common.PipelineMetrics]:
    """Full TTS pipeline using PyTorch fp32."""
    if opt is None:
        opt = common.OptConfig()
    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    metrics = common.PipelineMetrics(f"PyTorch fp32/CPU ({opt.solver}-{opt.diffusion_steps}s)")
    mem_before = common.get_peak_memory_mb()

    t0_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        common.MODEL_ID, torch_dtype=torch.float32,
    )
    model.eval()
    metrics.record("load", (time.perf_counter() - t0) * 1000)

    embed_table = model.get_input_embeddings().weight.detach()
    qwen = model.model.language_model
    lm_head = model.lm_head
    prediction_head = model.model.prediction_head
    acoustic_connector = model.model.acoustic_connector
    semantic_tokenizer = model.model.semantic_tokenizer
    semantic_connector = model.model.semantic_connector
    from traceable_modules import TraceableStreamingVAEDecoder
    vae_decoder = TraceableStreamingVAEDecoder(model.model.acoustic_tokenizer.decoder).eval()

    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
    semantic_tokenizer = semantic_tokenizer.to("mps")
    semantic_connector = semantic_connector.to("mps")
    sem_cache = VibeVoiceTokenizerStreamingCache()

    # Compute negative condition for CFG (unconditional = speech_start token only)
    with torch.no_grad():
        neg_embed = embed_table[torch.tensor([[common.SPEECH_START_ID]])]
        neg_out = qwen(inputs_embeds=neg_embed, use_cache=False)
    neg_condition_pt = neg_out.last_hidden_state[:, 0, :].numpy()

    # Prefill
    t0 = time.perf_counter()
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    inputs_embeds = embed_table[ids_tensor]

    # Inject voice cloning embeddings at speech_diffusion positions (per speaker)
    if voice_clone is not None:
        for spk in voice_clone.speakers:
            if spk._cached_embeds_pt is None:
                spk._cached_embeds_pt = common.encode_voice_reference_pt(
                    spk.ref_audio_np, model, spk.num_vae_tokens)
            speech_embeds_t = torch.tensor(spk._cached_embeds_pt, dtype=torch.float32)
            for i, pos in enumerate(spk.speech_embed_positions):
                if i < len(speech_embeds_t):
                    inputs_embeds[0, pos] = speech_embeds_t[i]

    # Run through LM to get KV cache
    with torch.no_grad():
        out = qwen(inputs_embeds=inputs_embeds, use_cache=True)
    past_kv = out.past_key_values
    last_hidden = out.last_hidden_state[:, -1:, :]
    metrics.record("prefill", (time.perf_counter() - t0) * 1000)
    metrics.num_text_tokens = len(input_ids)

    # Autoregressive generation
    audio_chunks = []
    rng = np.random.RandomState(seed)
    position = len(input_ids)

    # First token: get logits from last hidden of prefill
    with torch.no_grad():
        logits = lm_head(last_hidden)
    next_token = logits[0, 0].argmax().item()

    for step in range(max_speech_tokens * 3):  # generous upper bound
        if next_token == common.EOS_ID:
            break
        if metrics.num_speech_tokens >= max_speech_tokens:
            break

        if next_token == common.SPEECH_DIFFUSION_ID:
            metrics.num_speech_tokens += 1

            # Diffusion with CFG: use last hidden state to condition
            t0 = time.perf_counter()
            condition = last_hidden[:, 0, :].numpy()
            def pt_diffusion_fn(s, ts, c):
                with torch.no_grad():
                    return prediction_head(
                        torch.from_numpy(s), torch.from_numpy(ts),
                        condition=torch.from_numpy(c),
                    ).numpy()
            guided_fn = common.make_cfg_fn(pt_diffusion_fn, neg_condition_pt, opt.cfg_scale)
            sample = common._sample_latent(guided_fn, condition, opt, seed=rng.randint(0, 2**31))
            metrics.record("diffusion", (time.perf_counter() - t0) * 1000)

            # VAE decode
            t0 = time.perf_counter()
            latent = sample / common.SPEECH_SCALING_FACTOR - common.SPEECH_BIAS_FACTOR
            latent_t = torch.from_numpy(latent[:, :, None])  # (1, 64, 1)
            with torch.no_grad():
                audio = vae_decoder(latent_t).numpy()
            audio_chunks.append(audio.squeeze())
            metrics.record("vae", (time.perf_counter() - t0) * 1000)

            # Feed back: acoustic + streaming semantic connectors -> next embedding
            t0 = time.perf_counter()
            with torch.no_grad():
                acoustic_embed = acoustic_connector(torch.from_numpy(sample[:, None, :]))
                sem_embed = common._encode_semantic_streaming(
                    audio_chunks[-1], semantic_tokenizer, semantic_connector, sem_cache
                )
                next_embed = acoustic_embed + sem_embed
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            next_embed = embed_table[torch.tensor([[next_token]])]

        # LM step
        t0 = time.perf_counter()
        with torch.no_grad():
            out = qwen(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_hidden = out.last_hidden_state
        with torch.no_grad():
            logits = lm_head(last_hidden)
        next_token = logits[0, 0].argmax().item()
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000)
        position += 1

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    if audio_chunks:
        audio_out = np.concatenate(audio_chunks)
    else:
        audio_out = np.zeros(0, dtype=np.float32)
    metrics.audio_samples = len(audio_out)

    del model, past_kv
    return audio_out, metrics
