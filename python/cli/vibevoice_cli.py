#!/usr/bin/env python3
"""VibeVoice CoreML CLI — TTS and ASR inference using published HuggingFace models.

Downloads CoreML models automatically from https://huggingface.co/gafiatulin
and runs inference using coremltools.

Usage:
    uv run vibevoice-cli --model 0.5b --text "Hello world"
    uv run vibevoice-cli --model 7b --ref-audio speaker.wav --text "Hello from a cloned voice"
    uv run vibevoice-cli --model asr --audio recording.wav
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000

# HuggingFace repos
HF_REPOS = {
    "0.5b": "gafiatulin/vibevoice-tts-0.5b-coreml",
    "1.5b": "gafiatulin/vibevoice-tts-1.5b-coreml",
    "7b": "gafiatulin/vibevoice-tts-7b-coreml",
    "asr": "gafiatulin/vibevoice-asr-coreml",
}

MODEL_NAMES = {
    "0.5b": "0.5B Realtime",
    "1.5b": "1.5B Multispeaker",
    "7b": "7B Multispeaker",
    "asr": "ASR 7B",
}

# Architecture configs per model
ARCH = {
    "0.5b": {
        "hidden_size": 896, "head_dim": 64, "num_q_heads": 14, "num_kv_heads": 2,
        "base_lm_layers": 4, "tts_lm_layers": 20, "vocab_size": 151936,
        "vae_dim": 64, "semantic_dim": 128,
        "speech_scaling": 0.23339844, "speech_bias": -0.0703125,
        "default_cfg_scale": 1.5, "default_diffusion_steps": 5,
        "text_window_size": 5, "speech_window_size": 6,
        "tokenizer": "Qwen/Qwen2.5-0.5B",
    },
    "1.5b": {
        "hidden_size": 1536, "head_dim": 128, "num_q_heads": 12, "num_kv_heads": 2,
        "num_layers": 28, "vocab_size": 151936,
        "vae_dim": 64, "semantic_dim": 128,
        "speech_scaling": 0.1962890625, "speech_bias": -0.04931640625,
        "default_cfg_scale": 1.3, "default_diffusion_steps": 10,
        "tokenizer": "Qwen/Qwen2.5-1.5B",
    },
    "7b": {
        "hidden_size": 3584, "head_dim": 128, "num_q_heads": 28, "num_kv_heads": 4,
        "num_layers": 28, "vocab_size": 152064,
        "vae_dim": 64, "semantic_dim": 128,
        "speech_scaling": 0.1962890625, "speech_bias": -0.04931640625,
        "default_cfg_scale": 1.3, "default_diffusion_steps": 10,
        "tokenizer": "Qwen/Qwen2.5-7B",
    },
    "asr": {
        "hidden_size": 3584, "head_dim": 128, "num_q_heads": 28, "num_kv_heads": 4,
        "num_layers": 28, "vocab_size": 152064,
        "vae_dim": 64, "semantic_dim": 128,
        "hop_length": 3200, "vae_std": 0.625,
        "chunk_seconds": 60,
        "tokenizer": "microsoft/VibeVoice-ASR-HF",
    },
}

# Special token IDs (shared across models)
SPEECH_START_ID = 151652
SPEECH_END_ID = 151653
SPEECH_DIFFUSION_ID = 151654
EOS_ID = 151643
AUDIO_TOKEN_ID = 151648
AUDIO_BOS_ID = 151646
AUDIO_EOS_ID = 151647

ROPE_THETA = 1e6


# ─── Model download ──────────────────────────────────────────────────────────

def download_models(repo: str, cache_dir: Optional[str] = None,
                    verbose: bool = False) -> Path:
    """Download CoreML models from HuggingFace Hub, return local directory."""
    from huggingface_hub import snapshot_download

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if verbose:
        print(f"  Downloading {repo}...")
    local_dir = snapshot_download(repo, **kwargs)
    return Path(local_dir)


# ─── Shared utilities ─────────────────────────────────────────────────────────

def _model_path(models_dir: Path, name: str) -> Path:
    """Resolve a CoreML model path, trying .mlmodelc first then .mlpackage."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = models_dir / (name + ext)
        if p.exists():
            return p
    return models_dir / (name + ".mlmodelc")


class _CompiledModel:
    """Thin wrapper around _MLModelProxy for .mlmodelc files."""
    def __init__(self, proxy):
        from coremltools.models.model import MLState
        self._proxy = proxy
        self._MLState = MLState

    def predict(self, data: dict, state=None):
        st = None if state is None else state.__proxy__
        return self._proxy.predict(data, st)

    def make_state(self):
        return self._MLState(self._proxy.newState())


def _load_model_path(path: Path, compute_units: str = "ALL"):
    """Load a CoreML model from a resolved path (.mlmodelc or .mlpackage)."""
    import coremltools as ct

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    if path.suffix == ".mlpackage":
        cu = getattr(ct.ComputeUnit, compute_units)
        return ct.models.MLModel(str(path), compute_units=cu)

    from coremltools.libcoremlpython import _MLModelProxy
    return _CompiledModel(_MLModelProxy(str(path), compute_units, "", {}, None))


def _load_model(models_dir: Path, name: str, compute_units: str = "ALL"):
    """Load a CoreML model by name, resolving .mlmodelc/.mlpackage automatically."""
    return _load_model_path(_model_path(models_dir, name), compute_units)


def load_embeddings(path: Path) -> np.ndarray:
    """Load token embeddings from .bin (float16 with uint32 header) -> float32."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        vocab_size, hidden_size = int(header[0]), int(header[1])
        data = np.frombuffer(f.read(), dtype=np.float16)
    return data.reshape(vocab_size, hidden_size).astype(np.float32)


def compute_rope(position: int, head_dim: int):
    """Compute RoPE cos/sin for a single position. Returns (1, 1, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(half, dtype=np.float32) / half))
    angle = position * inv_freq
    cos_val = np.cos(angle).astype(np.float32)
    sin_val = np.sin(angle).astype(np.float32)
    return (np.concatenate([cos_val, cos_val]).reshape(1, 1, head_dim),
            np.concatenate([sin_val, sin_val]).reshape(1, 1, head_dim))


def compute_rope_batch(start: int, length: int, head_dim: int):
    """Compute RoPE cos/sin for a range. Returns (1, length, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(half, dtype=np.float64) / half))
    positions = np.arange(start, start + length, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    cos_val = np.cos(angles).astype(np.float32)
    sin_val = np.sin(angles).astype(np.float32)
    return (np.concatenate([cos_val, cos_val], axis=1).reshape(1, length, head_dim),
            np.concatenate([sin_val, sin_val], axis=1).reshape(1, length, head_dim))


def causal_mask(q: int, start_pos: int) -> np.ndarray:
    """Build causal attention mask: (1, 1, q, start_pos+q)."""
    total = start_pos + q
    mask = np.full((1, 1, q, total), -1e9, dtype=np.float32)
    for i in range(q):
        mask[0, 0, i, :start_pos + i + 1] = 0.0
    return mask


def write_wav(samples: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Write float32 samples to WAV file."""
    import soundfile as sf
    sf.write(path, samples, sample_rate)


# ─── Diffusion sampling ──────────────────────────────────────────────────────

DDPM_STEPS = 1000

_alphas_cumprod = None


def _get_alphas():
    global _alphas_cumprod
    if _alphas_cumprod is None:
        steps = np.arange(DDPM_STEPS + 1, dtype=np.float64) / DDPM_STEPS
        ac = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        _alphas_cumprod = (ac / ac[0])[:DDPM_STEPS].astype(np.float32)
    return _alphas_cumprod


def dpm_solver_sample(diffusion_fn, condition, num_steps: int = 10, seed=None):
    """DPM-Solver++ 2M for v-prediction with lower_order_final."""
    ac = _get_alphas().astype(np.float64)
    alpha = np.sqrt(ac)
    sigma = np.sqrt(1.0 - ac)
    lam = np.log(alpha / np.maximum(sigma, 1e-10))

    rng = np.random.RandomState(seed)
    t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)
    sample = rng.randn(1, 64).astype(np.float32)
    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        v = diffusion_fn(sample, np.array([float(s)], dtype=np.float32), condition)
        x0 = float(alpha[s]) * sample - float(sigma[s]) * v
        x0_list.append(x0)

        lam_s = float(lam[s])
        lam_t = float(lam[max(t, 0)])
        h = lam_t - lam_s

        is_last = (i == num_steps - 1)
        is_second_last = (i == num_steps - 2)
        lower_final = is_last and num_steps < 15
        lower_second = is_second_last and num_steps < 15
        use_first = len(x0_list) < 2 or lower_final or lower_second

        if use_first:
            D = x0_list[-1]
        else:
            s_prev = int(t_schedule[i - 1])
            h_prev = lam_s - float(lam[s_prev])
            r = h_prev / h
            D = x0_list[-1] + 0.5 * (1.0 / r) * (x0_list[-1] - x0_list[-2])

        sample = (float(sigma[t]) / float(sigma[s])) * sample \
               - float(alpha[t]) * float(np.expm1(-h)) * D

    return sample


# ─── 0.5B Streaming TTS ──────────────────────────────────────────────────────

def run_05b(models_dir: Path, text: str, voice: str, seed: int,
            output: str, verbose: bool):
    """Run 0.5B streaming TTS pipeline."""
    arch = ARCH["0.5b"]
    HS = arch["hidden_size"]
    HD = arch["head_dim"]
    VAE_DIM = arch["vae_dim"]

    print("Loading 0.5B models...")
    sys.stdout.flush()
    t0 = time.time()

    # Load CoreML models
    cml_base = _load_model(models_dir, "base_lm_stateful")
    cml_tts = _load_model(models_dir, "tts_lm_stateful")
    cml_vae = _load_model(models_dir, "vae_decoder_streaming", "CPU_AND_GPU")
    cml_eos = _load_model(models_dir, "eos_classifier")
    cml_conn = _load_model(models_dir, "acoustic_connector")

    # Load diffusion (B=2 batched if available)
    diff_b2_path = _model_path(models_dir, "diffusion_head_b2")
    diff_path = _model_path(models_dir, "diffusion_head")
    diff_loop_path = _model_path(models_dir, "diffusion_loop")
    cml_diff_loop = None
    cml_diff_b2 = None
    cml_diff = None
    if diff_loop_path.exists():
        cml_diff_loop = _load_model_path(diff_loop_path)
    elif diff_b2_path.exists():
        cml_diff_b2 = _load_model_path(diff_b2_path)
    elif diff_path.exists():
        cml_diff = _load_model_path(diff_path)

    # Embeddings
    embed_tokens = load_embeddings(models_dir / "embed_tokens.bin")
    tts_input_types = load_embeddings(models_dir / "tts_input_types.bin")

    # Load voice prompt (.vvvoice file)
    voice_data = _load_vvvoice(models_dir, voice)

    # Create states and inject KV caches
    base_state = cml_base.make_state()
    tts_state = cml_tts.make_state()
    neg_tts_state = cml_tts.make_state()
    vae_state = cml_vae.make_state()

    base_pos = _inject_kv_05b(cml_base, base_state, voice_data["lm"],
                              arch["base_lm_layers"], arch["num_kv_heads"], HD)
    tts_pos = _inject_kv_05b(cml_tts, tts_state, voice_data["tts_lm"],
                             arch["tts_lm_layers"], arch["num_kv_heads"], HD)
    neg_tts_pos = _inject_kv_05b(cml_tts, neg_tts_state, voice_data["neg_tts_lm"],
                                 arch["tts_lm_layers"], arch["num_kv_heads"], HD)

    tts_last_hidden = voice_data["tts_lm"]["last_hidden"]
    neg_tts_last_hidden = voice_data["neg_tts_lm"]["last_hidden"]

    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Pre-allocate zero injection tensors
    base_total_kv = arch["base_lm_layers"] * arch["num_kv_heads"]
    tts_total_kv = arch["tts_lm_layers"] * arch["num_kv_heads"]
    zero_inject = np.zeros((1,), dtype=np.float32)

    def base_predict(hidden, cos, sin, mask):
        q = hidden.shape[1]
        return cml_base.predict({
            "hidden_states": hidden.astype(np.float32),
            "position_cos": cos, "position_sin": sin,
            "attention_mask": mask,
            "inject_mode": zero_inject,
            "inject_k": np.zeros((1, base_total_kv, q, HD), dtype=np.float32),
            "inject_v": np.zeros((1, base_total_kv, q, HD), dtype=np.float32),
        }, state=base_state)

    def tts_predict(hidden, cos, sin, mask, state):
        q = hidden.shape[1]
        return cml_tts.predict({
            "hidden_states": hidden.astype(np.float32),
            "position_cos": cos, "position_sin": sin,
            "attention_mask": mask,
            "inject_mode": zero_inject,
            "inject_k": np.zeros((1, tts_total_kv, q, HD), dtype=np.float32),
            "inject_v": np.zeros((1, tts_total_kv, q, HD), dtype=np.float32),
        }, state=state)

    # Tokenize
    tts_text_ids = _tokenize_05b(text, arch["tokenizer"])
    text_window_idx = 0
    total_windows = math.ceil(len(tts_text_ids) / arch["text_window_size"])

    # Generation loop
    t1 = time.time()
    audio_chunks = []
    finished = False
    rng = np.random.RandomState(seed)
    cfg_scale = arch["default_cfg_scale"]
    diff_steps = arch["default_diffusion_steps"]
    speech_scaling = arch["speech_scaling"]
    speech_bias = arch["speech_bias"]

    while not finished:
        start = text_window_idx * arch["text_window_size"]
        end = (text_window_idx + 1) * arch["text_window_size"]
        cur_ids = tts_text_ids[start:end]
        text_window_idx += 1

        if len(cur_ids) > 0:
            embeds = embed_tokens[cur_ids][None, :, :]
            q = len(cur_ids)

            cos, sin = compute_rope_batch(base_pos, q, HD)
            mask = causal_mask(q, base_pos)
            lm_out = base_predict(embeds, cos, sin, mask)
            lm_hidden = lm_out["output_hidden"]
            base_pos += q

            tts_embeds = lm_hidden + tts_input_types[1:2, :]
            cos, sin = compute_rope_batch(tts_pos, q, HD)
            mask = causal_mask(q, tts_pos)
            tts_out = tts_predict(tts_embeds, cos, sin, mask, tts_state)
            tts_last_hidden = tts_out["output_hidden"][:, -1:, :]
            tts_pos += q

        for _ in range(arch["speech_window_size"]):
            pos_cond = tts_last_hidden[:, 0, :]
            neg_cond = neg_tts_last_hidden[:, 0, :]

            # Diffusion
            if cml_diff_loop is not None:
                inner_seed = rng.randint(0, 2**31)
                noise = np.random.RandomState(inner_seed).randn(1, VAE_DIM).astype(np.float32)
                latent = cml_diff_loop.predict({
                    "noise": noise, "condition": pos_cond,
                    "neg_condition": neg_cond,
                    "cfg_scale": np.array([cfg_scale], dtype=np.float32),
                })["latent"]
            elif cml_diff_b2 is not None:
                def diff_fn_b2(s, t, c):
                    s_b2 = np.concatenate([s, s], axis=0)
                    t_b2 = np.concatenate([t, t], axis=0)
                    c_b2 = np.concatenate([c, neg_cond.reshape(1, -1)], axis=0)
                    v = cml_diff_b2.predict({
                        "noisy_latent": s_b2, "timestep": t_b2, "condition": c_b2,
                    })["predicted_noise"]
                    return v[0:1] + cfg_scale * (v[0:1] - v[1:2]) if cfg_scale > 1.0 else v[0:1]

                # Correct CFG: v_uncond + scale * (v_cond - v_uncond)
                def guided_fn_b2(s, t, c):
                    s_b2 = np.concatenate([s, s], axis=0)
                    t_b2 = np.concatenate([t, t], axis=0)
                    c_b2 = np.concatenate([c, neg_cond.reshape(1, -1)], axis=0)
                    v = cml_diff_b2.predict({
                        "noisy_latent": s_b2.astype(np.float32),
                        "timestep": t_b2.astype(np.float32),
                        "condition": c_b2.astype(np.float32),
                    })["predicted_noise"]
                    v_cond = v[0:1]
                    v_uncond = v[1:2]
                    return v_uncond + cfg_scale * (v_cond - v_uncond)

                latent = dpm_solver_sample(
                    guided_fn_b2, pos_cond.reshape(1, -1), diff_steps,
                    seed=rng.randint(0, 2**31),
                )
            else:
                def diff_fn(s, t, c):
                    return cml_diff.predict({
                        "noisy_latent": s.astype(np.float32),
                        "timestep": t.astype(np.float32),
                        "condition": c.astype(np.float32),
                    })["predicted_noise"]

                def guided_fn(s, t, c):
                    v_cond = diff_fn(s, t, c)
                    v_uncond = diff_fn(s, t, neg_cond.reshape(1, -1))
                    return v_uncond + cfg_scale * (v_cond - v_uncond)

                latent = dpm_solver_sample(
                    guided_fn if cfg_scale > 1.0 else diff_fn,
                    pos_cond.reshape(1, -1), diff_steps,
                    seed=rng.randint(0, 2**31),
                )

            # VAE decode
            scaled = latent / speech_scaling - speech_bias
            audio = cml_vae.predict(
                {"latent": scaled.T[None, :, :].astype(np.float32)},
                state=vae_state,
            )["audio"]
            audio_chunks.append(audio.flatten())

            # Connector feedback
            embed = cml_conn.predict({
                "speech_latent": latent[:, None, :].astype(np.float32),
            })["embedding"]

            # TTS LM with speech embedding
            tts_embeds = embed + tts_input_types[0:1, :]
            cos, sin = compute_rope(tts_pos, HD)
            mask = np.zeros((1, 1, 1, tts_pos + 1), dtype=np.float32)
            tts_out = tts_predict(tts_embeds, cos, sin, mask, tts_state)
            tts_last_hidden = tts_out["output_hidden"][:, -1:, :]
            tts_pos += 1

            # Negative TTS LM
            cos, sin = compute_rope(neg_tts_pos, HD)
            mask = np.zeros((1, 1, 1, neg_tts_pos + 1), dtype=np.float32)
            neg_out = tts_predict(embed + tts_input_types[0:1, :], cos, sin, mask, neg_tts_state)
            neg_tts_last_hidden = neg_out["output_hidden"][:, -1:, :]
            neg_tts_pos += 1

            # EOS check
            eos_prob = cml_eos.predict({
                "hidden_state": tts_last_hidden[:, 0, :].astype(np.float32),
            })["eos_probability"]
            if float(eos_prob.flatten()[0]) > 0.5:
                finished = True
                break

        max_frames = max(len(tts_text_ids) * 5, 30)
        if len(audio_chunks) >= max_frames:
            break
        if text_window_idx >= total_windows and not finished:
            if text_window_idx > total_windows + 10:
                break

    gen_time = time.time() - t1
    if audio_chunks:
        audio_out = np.concatenate(audio_chunks)
    else:
        audio_out = np.zeros(SAMPLE_RATE, dtype=np.float32)

    duration = len(audio_out) / SAMPLE_RATE
    rtf = duration / gen_time if gen_time > 0 else 0
    print(f"Generated {len(audio_chunks)} frames, {duration:.1f}s audio "
          f"in {gen_time:.1f}s ({rtf:.2f}x RTF)")

    write_wav(audio_out, output)
    print(f"Saved to {output}")


def _tokenize_05b(text: str, tokenizer_name: str):
    """Tokenize text for 0.5B TTS."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tok.encode(text.strip() + "\n", add_special_tokens=False)


def _load_vvvoice(models_dir: Path, voice_name: str) -> dict:
    """Load a .vvvoice voice prompt file.

    Format: binary file with 4 KV cache sections (lm, tts_lm, neg_lm, neg_tts_lm),
    each containing k/v caches and last_hidden_state.
    """
    voices_dir = models_dir / "voices"
    if not voices_dir.exists():
        voices_dir = models_dir

    # Find matching voice file
    candidates = list(voices_dir.glob("*.vvvoice"))
    voice_path = None
    for c in candidates:
        name = c.stem.split("-")[-1].split("_")[0]
        if name.lower() == voice_name.lower():
            voice_path = c
            break

    if voice_path is None and candidates:
        print(f"  Warning: voice '{voice_name}' not found, using {candidates[0].stem}")
        voice_path = candidates[0]

    if voice_path is None:
        raise FileNotFoundError(f"No voice files found in {voices_dir}")

    return _parse_vvvoice(voice_path)


def _parse_vvvoice(path: Path) -> dict:
    """Parse .vvvoice binary format.

    Header (8 bytes): magic(4 "VVVP") version(uint16) num_sections(uint16)
    Section table (num_sections x 16 bytes each):
        num_layers(uint16) num_kv_heads(uint16) seq_len(uint32)
        head_dim(uint16) hidden_dim(uint16) data_offset(uint32)
    Data at data_offset:
        k_cache: float16[total_kv * seq_len * head_dim]
        v_cache: float16[same]
        last_hidden: float16[hidden_dim]
    """
    data = path.read_bytes()
    magic = data[:4]
    if magic != b"VVVP":
        raise ValueError(f"Invalid .vvvoice magic: {magic}")

    version, num_sections = struct.unpack_from("<HH", data, 4)
    if version != 1:
        raise ValueError(f"Unsupported .vvvoice version: {version}")

    section_names = ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]
    sections = {}

    for i in range(num_sections):
        entry_offset = 8 + i * 16
        num_layers, num_kv_heads, seq_len, head_dim, hidden_dim, data_offset = \
            struct.unpack_from("<HHIHHI", data, entry_offset)

        total_kv = num_layers * num_kv_heads
        kv_elems = total_kv * seq_len * head_dim
        kv_bytes = kv_elems * 2  # float16

        k = np.frombuffer(data, dtype=np.float16, count=kv_elems,
                          offset=data_offset).copy()
        k = k.reshape(total_kv, seq_len, head_dim).astype(np.float32)

        v = np.frombuffer(data, dtype=np.float16, count=kv_elems,
                          offset=data_offset + kv_bytes).copy()
        v = v.reshape(total_kv, seq_len, head_dim).astype(np.float32)

        h = np.frombuffer(data, dtype=np.float16, count=hidden_dim,
                          offset=data_offset + 2 * kv_bytes).copy()
        last_hidden = h.astype(np.float32).reshape(1, 1, hidden_dim)

        name = section_names[i] if i < len(section_names) else f"section_{i}"
        sections[name] = {
            "k": k, "v": v,
            "last_hidden": last_hidden,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    return sections


def _inject_kv_05b(cml_model, state, section: dict,
                   num_layers: int, num_kv_heads: int, head_dim: int) -> int:
    """Inject KV cache from .vvvoice section into CoreML model state."""
    seq_len = section["seq_len"]
    total_kv = num_layers * num_kv_heads

    # .vvvoice stores (total_kv, seq_len, head_dim) already concatenated
    k = section["k"]  # (total_kv, seq_len, head_dim)
    v = section["v"]

    inject_k = k.reshape(1, total_kv, seq_len, head_dim)
    inject_v = v.reshape(1, total_kv, seq_len, head_dim)

    cos, sin = compute_rope_batch(0, seq_len, head_dim)
    mask = causal_mask(seq_len, 0)
    dummy_hidden = np.zeros((1, seq_len, section["last_hidden"].shape[-1]), dtype=np.float32)

    cml_model.predict({
        "hidden_states": dummy_hidden,
        "position_cos": cos, "position_sin": sin,
        "attention_mask": mask,
        "inject_mode": np.array([1.0], dtype=np.float32),
        "inject_k": inject_k,
        "inject_v": inject_v,
    }, state=state)

    return seq_len


# ─── 1.5B / 7B Multispeaker TTS ──────────────────────────────────────────────

def run_multispeaker(models_dir: Path, model_key: str, text: str,
                     ref_audio: list, seed: int, max_tokens: Optional[int],
                     output: str, verbose: bool):
    """Run 1.5B or 7B multispeaker TTS pipeline."""
    arch = ARCH[model_key]
    HS = arch["hidden_size"]
    HD = arch["head_dim"]
    VAE_DIM = arch["vae_dim"]
    cfg_scale = arch["default_cfg_scale"]

    print(f"Loading {MODEL_NAMES[model_key]} models...")
    sys.stdout.flush()
    t0 = time.time()

    # Load CoreML models — fused int8 LM preferred
    lm_path = _model_path(models_dir, "lm_decoder_fused_int8")
    if not lm_path.exists():
        for name in ["lm_decoder_fused", "lm_decoder_stateful_int8",
                      "lm_decoder_stateful"]:
            p = _model_path(models_dir, name)
            if p.exists():
                lm_path = p
                break
    fused = "fused" in lm_path.name

    cml_lm = _load_model_path(lm_path, "CPU_AND_GPU")

    # Diffusion: fused loop preferred
    diff_loop_path = _model_path(models_dir, "diffusion_loop")
    diff_head_path = _model_path(models_dir, "diffusion_head")
    cml_diff_loop = None
    cml_diff = None
    if diff_loop_path.exists():
        cml_diff_loop = _load_model_path(diff_loop_path)
    elif diff_head_path.exists():
        cml_diff = _load_model_path(diff_head_path)

    cml_vae = _load_model(models_dir, "vae_decoder_streaming", "CPU_AND_GPU")
    vae_state = cml_vae.make_state()

    cml_conn = _load_model(models_dir, "acoustic_connector")
    cml_head = None if fused else _load_model(models_dir, "lm_head")

    # Semantic encoder + connector
    cml_sem_enc = _load_model(models_dir, "semantic_encoder_streaming", "CPU_AND_GPU")
    cml_sem_conn = _load_model(models_dir, "semantic_connector")
    sem_state = cml_sem_enc.make_state()

    embed_table = load_embeddings(models_dir / "embed_tokens.bin")

    # Voice cloning setup
    cml_vae_enc = None
    if ref_audio:
        vae_enc_path = _model_path(models_dir, "vae_encoder")
        if vae_enc_path.exists():
            cml_vae_enc = _load_model_path(vae_enc_path)

    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Tokenize prompt
    input_ids, voice_embeds = _tokenize_multispeaker(
        text, ref_audio, arch, cml_vae_enc, cml_conn,
    )

    # Estimate max tokens
    import re
    if max_tokens is None:
        text_only = re.sub(r"Speaker\s+\d+\s*:", "", text)
        word_count = len(text_only.split())
        max_tokens = max(20, int(word_count * 4.5))

    print(f'Generating: "{text}"')
    sys.stdout.flush()

    # Warmup
    warmup_state = cml_lm.make_state()
    dummy_h = np.zeros((1, 1, HS), dtype=np.float32)
    dummy_cos = np.ones((1, 1, HD), dtype=np.float32)
    dummy_sin = np.zeros((1, 1, HD), dtype=np.float32)
    dummy_mask = np.zeros((1, 1, 1, 1), dtype=np.float32)
    cml_lm.predict({"hidden_states": dummy_h, "position_cos": dummy_cos,
                     "position_sin": dummy_sin, "attention_mask": dummy_mask},
                    state=warmup_state)
    del warmup_state

    # Negative condition for CFG
    neg_state = cml_lm.make_state()
    neg_embed = embed_table[SPEECH_START_ID][None, None, :]
    neg_cos, neg_sin = compute_rope(0, HD)
    neg_out = cml_lm.predict({
        "hidden_states": neg_embed.astype(np.float32),
        "position_cos": neg_cos, "position_sin": neg_sin,
        "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }, state=neg_state)
    neg_condition = neg_out["output_hidden"][:, 0:1, :].reshape(1, HS)
    del neg_state

    # Prefill
    state = cml_lm.make_state()
    seq_len = len(input_ids)
    embeds = embed_table[input_ids][None, :, :]

    # Inject voice clone embeddings
    if voice_embeds:
        for positions, emb_array in voice_embeds:
            for i, pos in enumerate(positions):
                if i < len(emb_array):
                    embeds[0, pos] = emb_array[i]

    cos, sin = compute_rope_batch(0, seq_len, HD)
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    mask = np.triu(np.full_like(mask, -1e9), k=1)

    lm_out = cml_lm.predict({
        "hidden_states": embeds, "position_cos": cos, "position_sin": sin,
        "attention_mask": mask,
    }, state=state)

    if fused:
        hidden = lm_out["output_hidden"][:, -1:, :]
        next_token = int(np.argmax(lm_out["logits"][0, -1]))
    else:
        hidden = lm_out["output_hidden"][:, -1:, :]
        logits = cml_head.predict({"hidden_state": hidden})["logits"]
        next_token = int(np.argmax(logits[0, 0]))

    # Guided diffusion function
    def guided_diffusion(sample, timestep, condition):
        return cml_diff.predict({
            "noisy_latent": sample.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "condition": condition.astype(np.float32),
        })["predicted_noise"]

    def make_guided(neg_c):
        def fn(s, t, c):
            v_cond = guided_diffusion(s, t, c)
            v_uncond = guided_diffusion(s, t, neg_c)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        return fn

    # Autoregressive generation
    t1 = time.time()
    audio_chunks = []
    rng = np.random.RandomState(seed)
    position = seq_len
    speech_tokens = 0

    for step in range(max_tokens * 3):
        if next_token == EOS_ID:
            break
        if speech_tokens >= max_tokens:
            break

        if next_token == SPEECH_DIFFUSION_ID:
            speech_tokens += 1
            condition = hidden[:, 0:1, :].reshape(1, HS)

            if cml_diff_loop is not None:
                inner_seed = rng.randint(0, 2**31)
                noise = np.random.RandomState(inner_seed).randn(1, VAE_DIM).astype(np.float32)
                sample = cml_diff_loop.predict({
                    "noise": noise, "condition": condition,
                    "neg_condition": neg_condition,
                    "cfg_scale": np.array([cfg_scale], dtype=np.float32),
                })["latent"]
            else:
                guided_fn = make_guided(neg_condition)
                sample = dpm_solver_sample(
                    guided_fn, condition,
                    arch["default_diffusion_steps"],
                    seed=rng.randint(0, 2**31),
                )

            # VAE decode
            latent = sample / arch["speech_scaling"] - arch["speech_bias"]
            audio = cml_vae.predict(
                {"latent": latent[:, :, None].astype(np.float32)},
                state=vae_state,
            )["audio"]
            audio_chunks.append(audio.squeeze())

            # Acoustic + semantic feedback
            acoustic_embed = cml_conn.predict({
                "speech_latent": sample[:, None, :].astype(np.float32),
            })["embedding"]

            if audio_chunks:
                chunk = audio_chunks[-1][:3200].astype(np.float32)
                audio_input = np.zeros((1, 1, 3200), dtype=np.float32)
                audio_input[0, 0, :len(chunk)] = chunk
                features = cml_sem_enc.predict(
                    {"audio": audio_input}, state=sem_state
                )["features"]
                feat = features.transpose(0, 2, 1)
                sem_embed = cml_sem_conn.predict(
                    {"semantic_features": feat}
                )["embedding"]
                next_embed = acoustic_embed + sem_embed
            else:
                next_embed = acoustic_embed

        else:
            if next_token == SPEECH_END_ID:
                vae_state = cml_vae.make_state()
                sem_state = cml_sem_enc.make_state()
            next_embed = embed_table[next_token][None, None, :]

        # LM step
        cos, sin = compute_rope(position, HD)
        mask = np.zeros((1, 1, 1, position + 1), dtype=np.float32)
        lm_out = cml_lm.predict({
            "hidden_states": next_embed.astype(np.float32),
            "position_cos": cos, "position_sin": sin,
            "attention_mask": mask,
        }, state=state)

        if fused:
            hidden = lm_out["output_hidden"]
            next_token = int(np.argmax(lm_out["logits"][0, 0]))
        else:
            hidden = lm_out["output_hidden"]
            logits = cml_head.predict({"hidden_state": hidden})["logits"]
            next_token = int(np.argmax(logits[0, 0]))
        position += 1

    gen_time = time.time() - t1
    if audio_chunks:
        audio_out = np.concatenate(audio_chunks)
    else:
        audio_out = np.zeros(0, dtype=np.float32)

    duration = len(audio_out) / SAMPLE_RATE
    rtf = duration / gen_time if gen_time > 0 else 0
    print(f"Generated {speech_tokens} speech tokens, {duration:.1f}s audio "
          f"in {gen_time:.1f}s ({rtf:.2f}x RTF)")

    write_wav(audio_out, output)
    print(f"Saved to {output}")


def _tokenize_multispeaker(text: str, ref_audio: list, arch: dict,
                           cml_vae_enc, cml_conn) -> tuple:
    """Build prompt tokens for multispeaker TTS. Returns (input_ids, voice_embeds)."""
    import re
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(arch["tokenizer"], trust_remote_code=True)
    system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    system_tokens = tok.encode(system_prompt)

    voice_embeds = []

    if ref_audio:
        voice_prefix = tok.encode(" Voice input:\n", add_special_tokens=False)
        newline_tok = tok.encode("\n", add_special_tokens=False)
        voice_tokens = list(voice_prefix)
        current_offset = len(system_tokens) + len(voice_prefix)

        for spk_idx, audio_path in enumerate(ref_audio):
            wav = _load_audio_for_cloning(audio_path)
            num_vae_tokens = math.ceil(len(wav) / 3200)

            spk_prefix = tok.encode(f" Speaker {spk_idx}:", add_special_tokens=False)
            voice_tokens += spk_prefix
            current_offset += len(spk_prefix)

            voice_tokens.append(SPEECH_START_ID)
            current_offset += 1

            speech_positions = list(range(current_offset, current_offset + num_vae_tokens))
            voice_tokens += [SPEECH_DIFFUSION_ID] * num_vae_tokens
            current_offset += num_vae_tokens

            voice_tokens.append(SPEECH_END_ID)
            voice_tokens += newline_tok
            current_offset += 1 + len(newline_tok)

            # Encode voice reference
            if cml_vae_enc is not None:
                emb = _encode_voice_coreml(wav, num_vae_tokens, cml_vae_enc, cml_conn, arch)
                voice_embeds.append((speech_positions, emb))

        text_section = tok.encode(" Text input:\n", add_special_tokens=False)
        lines = text.strip().split("\n")
        speaker_tokens = []
        for line in lines:
            def _dec(m):
                return f"Speaker {max(0, int(m.group(1)) - 1)}"
            line_clean = re.sub(r"Speaker\s+(\d+)", _dec, line.strip())
            speaker_tokens += tok.encode(f" {line_clean}\n", add_special_tokens=False)

        output_section = tok.encode(" Speech output:\n", add_special_tokens=False)
        all_ids = (system_tokens + voice_tokens + text_section +
                   speaker_tokens + output_section + [SPEECH_START_ID])
        return all_ids, voice_embeds

    # No voice cloning
    text_section = tok.encode(" Text input:\n", add_special_tokens=False)
    lines = text.strip().split("\n")
    speaker_tokens = []
    for line in lines:
        speaker_tokens += tok.encode(f" {line.strip()}\n", add_special_tokens=False)
    output_section = tok.encode(" Speech output:\n", add_special_tokens=False)
    return (system_tokens + text_section + speaker_tokens +
            output_section + [SPEECH_START_ID]), []


def _load_audio_for_cloning(audio_path: str) -> np.ndarray:
    """Load audio, resample to 24kHz mono, cap at 10s."""
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        gcd = math.gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
    max_samples = SAMPLE_RATE * 10
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    return wav.astype(np.float32)


def _encode_voice_coreml(wav: np.ndarray, num_vae_tokens: int,
                         cml_vae_enc, cml_conn, arch: dict) -> np.ndarray:
    """Encode reference audio to speech embeddings using CoreML."""
    max_samples = SAMPLE_RATE * 10
    audio_input = np.zeros((1, 1, max_samples), dtype=np.float32)
    actual_len = min(len(wav), max_samples)
    audio_input[0, 0, :actual_len] = wav[:actual_len]

    latents = cml_vae_enc.predict({"audio": audio_input})["latent"]
    actual_t = min(latents.shape[2], num_vae_tokens)
    latents = latents[:, :, :actual_t]

    features = (latents + arch["speech_bias"]) * arch["speech_scaling"]
    embeds = []
    for t in range(actual_t):
        frame = features[:, :, t:t+1].transpose(0, 2, 1)
        emb = cml_conn.predict({"speech_latent": frame})["embedding"]
        embeds.append(emb[0, 0])
    return np.stack(embeds)


# ─── ASR Pipeline ─────────────────────────────────────────────────────────────

def run_asr(models_dir: Path, audio_path: str, prompt: Optional[str],
            max_tokens: Optional[int], seed: int, output: str, verbose: bool):
    """Run ASR (speech-to-text) pipeline."""
    arch = ARCH["asr"]
    HS = arch["hidden_size"]
    HD = arch["head_dim"]
    HOP = arch["hop_length"]

    print("Loading ASR models...")
    sys.stdout.flush()
    t0 = time.time()

    # Load encoders (fused preferred)
    fused_enc_path = _model_path(models_dir, "fused_encoder")
    if fused_enc_path.exists():
        cml_enc = _load_model_path(fused_enc_path, "CPU_AND_GPU")
        use_fused_enc = True
    else:
        cml_ac_enc = _load_model(models_dir, "acoustic_encoder", "CPU_AND_GPU")
        cml_sem_enc = _load_model(models_dir, "semantic_encoder", "CPU_AND_GPU")
        use_fused_enc = False

    # Projectors (fused preferred)
    fused_proj_path = _model_path(models_dir, "fused_projector")
    if fused_proj_path.exists():
        cml_proj = _load_model_path(fused_proj_path)
        use_fused_proj = True
    else:
        cml_ac_proj = _load_model(models_dir, "acoustic_projector")
        cml_sem_proj = _load_model(models_dir, "semantic_projector")
        use_fused_proj = False

    # LM decoder (fused int8 preferred)
    lm_path = _model_path(models_dir, "lm_decoder_fused_int8")
    if not lm_path.exists():
        for name in ["lm_decoder_fused", "lm_decoder_stateful_int8",
                      "lm_decoder_stateful"]:
            p = _model_path(models_dir, name)
            if p.exists():
                lm_path = p
                break
    fused = "fused" in lm_path.name

    cml_lm = _load_model_path(lm_path, "CPU_AND_GPU")
    lm_state = cml_lm.make_state()

    cml_head = None if fused else _load_model(models_dir, "lm_head")
    embed_table = load_embeddings(models_dir / "embed_tokens.bin")

    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load and preprocess audio
    wav = _load_audio_asr(audio_path)
    duration_secs = len(wav) / SAMPLE_RATE
    num_audio_tokens = math.ceil(len(wav) / HOP)
    print(f"Audio: {duration_secs:.1f}s, {num_audio_tokens} tokens")

    # Encode audio in 60s chunks
    chunk_samples = SAMPLE_RATE * arch["chunk_seconds"]
    chunk_tokens = chunk_samples // HOP
    num_chunks = math.ceil(len(wav) / chunk_samples)

    acoustic_list = []
    semantic_list = []

    for c in range(num_chunks):
        start = c * chunk_samples
        end = min(start + chunk_samples, len(wav))
        chunk = np.zeros(chunk_samples, dtype=np.float32)
        chunk[:end - start] = wav[start:end]
        audio_input = chunk.reshape(1, 1, chunk_samples)

        if use_fused_enc:
            out = cml_enc.predict({"audio": audio_input})
            ac_out = out["acoustic_features"]
            sem_out = out["semantic_features"]
        else:
            ac_out = cml_ac_enc.predict({"audio": audio_input})["features"]
            sem_out = cml_sem_enc.predict({"audio": audio_input})["features"]

        actual_tokens = min(chunk_tokens, math.ceil((end - start) / HOP))
        acoustic_list.append(ac_out[:, :actual_tokens, :])
        semantic_list.append(sem_out[:, :actual_tokens, :])

    acoustic = np.concatenate(acoustic_list, axis=1)
    semantic = np.concatenate(semantic_list, axis=1)

    # VAE noise injection
    rng = np.random.RandomState(42)
    noise_std = arch["vae_std"] * rng.randn(1).astype(np.float32)
    acoustic = acoustic + noise_std[:, None, None] * rng.randn(*acoustic.shape).astype(np.float32)

    # Project to LM space
    if use_fused_proj:
        audio_embeddings = cml_proj.predict({
            "acoustic_features": acoustic.astype(np.float32),
            "semantic_features": semantic.astype(np.float32),
        })["embedding"].astype(np.float32)
    else:
        ac_emb = cml_ac_proj.predict({"features": acoustic.astype(np.float32)})["embedding"]
        sem_emb = cml_sem_proj.predict({"features": semantic.astype(np.float32)})["embedding"]
        audio_embeddings = (ac_emb + sem_emb).astype(np.float32)

    # Build prompt
    input_ids = _build_asr_prompt(num_audio_tokens, duration_secs, prompt, arch)
    if max_tokens is None:
        max_tokens = max(256, int(duration_secs * 12))

    # Prefill
    ids_array = np.array(input_ids, dtype=np.int32)
    all_embeds = embed_table[ids_array]
    audio_positions = np.where(ids_array == AUDIO_TOKEN_ID)[0]
    num_audio = min(len(audio_positions), audio_embeddings.shape[1])
    all_embeds[audio_positions[:num_audio]] = audio_embeddings[0, :num_audio, :]
    all_embeds = all_embeds.reshape(1, len(input_ids), HS).astype(np.float32)

    seq_len = all_embeds.shape[1]
    cos, sin = compute_rope_batch(0, seq_len, HD)
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        mask[0, 0, i, i + 1:] = -1e9

    print("Prefilling...")
    sys.stdout.flush()
    out = cml_lm.predict({
        "hidden_states": all_embeds,
        "position_cos": cos, "position_sin": sin,
        "attention_mask": mask,
    }, state=lm_state)

    if fused:
        last_logits = out["logits"][:, -1:, :]
    else:
        last_hidden = out["output_hidden"][:, -1:, :]
        last_logits = cml_head.predict(
            {"hidden_state": last_hidden.astype(np.float32)}
        )["logits"]

    # Precompute RoPE table
    max_pos = seq_len + max_tokens
    rope_cos, rope_sin = compute_rope_batch(0, max_pos, HD)
    mask_buf = np.zeros((1, 1, 1, max_pos + 1), dtype=np.float32)

    # Autoregressive generation
    print("Generating...")
    sys.stdout.flush()
    t1 = time.time()
    next_token = int(np.argmax(last_logits[0, 0]))
    generated_ids = []
    position = seq_len

    for step in range(max_tokens):
        if next_token == EOS_ID:
            break

        generated_ids.append(next_token)

        hidden = embed_table[next_token:next_token + 1].reshape(1, 1, HS).astype(np.float32)
        cos = rope_cos[:, position:position + 1, :]
        sin = rope_sin[:, position:position + 1, :]
        mask = mask_buf[:, :, :, :position + 1]

        out = cml_lm.predict({
            "hidden_states": hidden,
            "position_cos": cos, "position_sin": sin,
            "attention_mask": mask,
        }, state=lm_state)

        if fused:
            logits = out["logits"]
        else:
            logits = cml_head.predict(
                {"hidden_state": out["output_hidden"].astype(np.float32)}
            )["logits"]

        next_token = int(np.argmax(logits[0, 0]))
        position += 1

    gen_time = time.time() - t1
    tok_per_sec = len(generated_ids) / gen_time if gen_time > 0 else 0
    print(f"Generated {len(generated_ids)} tokens in {gen_time:.1f}s "
          f"({tok_per_sec:.1f} tok/s)")

    # Decode
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(arch["tokenizer"], trust_remote_code=True)
    transcription = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Write output
    out_path = output if output else _default_asr_output(audio_path)
    Path(out_path).write_text(transcription)
    print(f"Saved to {out_path}")
    print(f"\n{transcription}")


def _load_audio_asr(audio_path: str) -> np.ndarray:
    """Load audio for ASR: mono 24kHz, normalized to -25 dB FS, padded."""
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        gcd = math.gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)

    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 0:
        target_rms = 10 ** (-25 / 20)
        wav = wav * (target_rms / rms)

    hop = ARCH["asr"]["hop_length"]
    remainder = len(wav) % hop
    if remainder > 0:
        wav = np.pad(wav, (0, hop - remainder))
    return wav.astype(np.float32)


def _build_asr_prompt(num_audio_tokens: int, duration_secs: float,
                      prompt: Optional[str], arch: dict) -> list:
    """Build ASR prompt token sequence."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(arch["tokenizer"], trust_remote_code=True)

    system_msg = "You are a helpful assistant that transcribes audio input into text output in JSON format."
    nl = tok.encode("\n", add_special_tokens=False)[0]
    system_ids = tok.encode(f"system\n{system_msg}", add_special_tokens=False)
    user_prefix = tok.encode("user\n", add_special_tokens=False)
    assistant_ids = tok.encode("assistant\n", add_special_tokens=False)

    IM_START = 151644
    IM_END = 151645

    prefix = [IM_START] + system_ids + [IM_END, nl, IM_START] + user_prefix
    suffix = [IM_START] + assistant_ids

    audio_section = [AUDIO_BOS_ID] + [AUDIO_TOKEN_ID] * num_audio_tokens + [AUDIO_EOS_ID]

    if prompt:
        user_text = (f"This is a {duration_secs:.2f} seconds audio, with extra info: {prompt}\n"
                     f"Please transcribe it with these keys: Start time, End time, Speaker ID, Content")
    else:
        user_text = (f"This is a {duration_secs:.2f} seconds audio, "
                     f"please transcribe it with these keys: Start time, End time, Speaker ID, Content")

    user_text_ids = tok.encode(f"\n{user_text}", add_special_tokens=False)

    return prefix + audio_section + user_text_ids + [IM_END, nl] + suffix


def _default_asr_output(audio_path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = Path(audio_path).stem
    return f"{ts}_{name}_coreml.txt"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice CoreML CLI — TTS and ASR inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  vibevoice-cli --model 0.5b --text "Hello world"
  vibevoice-cli --model 7b --ref-audio spk1.wav --text "Hello from a cloned voice"
  vibevoice-cli --model asr --audio recording.wav
  vibevoice-cli --model 0.5b --models-dir /path/to/models --text "Use local models"
""")
    parser.add_argument("--model", default=None,
                        choices=list(HF_REPOS.keys()),
                        help="Model variant: 0.5b, 1.5b, 7b, or asr")
    parser.add_argument("--models-dir", default=None,
                        help="Local models directory (default: auto-download from HuggingFace)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory for downloaded models")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed logs")

    # TTS options
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # 0.5B only
    parser.add_argument("--voice", default="Emma", help="Voice name (0.5B only, default: Emma)")

    # 1.5B/7B only
    parser.add_argument("--ref-audio", nargs="+", default=None,
                        help="Reference audio file(s) for voice cloning (1.5B/7B only)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum speech/generation tokens (default: auto)")

    # ASR options
    parser.add_argument("--audio", default=None, help="Input audio file for ASR")
    parser.add_argument("--prompt", default=None, help="Optional ASR context prompt")

    args = parser.parse_args()

    if args.model is None:
        parser.print_help()
        sys.exit(0)

    # Validate required args before downloading
    if args.model == "asr" and not args.audio:
        parser.error("--audio is required for ASR mode")
    if args.model != "asr" and not args.text:
        parser.error("--text is required for TTS mode")

    # Resolve models directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        repo = HF_REPOS[args.model]
        print(f"Syncing {MODEL_NAMES[args.model]} from {repo}...")
        sys.stdout.flush()
        models_dir = download_models(repo, cache_dir=args.cache_dir, verbose=args.verbose)
        print(f"Models ready: {models_dir}")

    # Default output
    if args.output is None:
        if args.model == "asr":
            args.output = ""  # will be auto-generated
        else:
            args.output = "output.wav"

    # Dispatch
    if args.model == "asr":
        run_asr(models_dir, args.audio, args.prompt, args.max_tokens,
                args.seed, args.output, args.verbose)
    elif args.model == "0.5b":
        run_05b(models_dir, args.text, args.voice, args.seed,
                args.output, args.verbose)
    else:
        run_multispeaker(models_dir, args.model, args.text,
                         args.ref_audio or [], args.seed, args.max_tokens,
                         args.output, args.verbose)


if __name__ == "__main__":
    main()
