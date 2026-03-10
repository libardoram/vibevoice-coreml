"""Shared constants, configs, metrics, and audio preprocessing for ASR pipeline."""

from __future__ import annotations

import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# ─── Architecture constants ──────────────────────────────────────────────────

HIDDEN_SIZE = 3584
HEAD_DIM = 128
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
NUM_LAYERS = 28
VOCAB_SIZE = 152064
RMS_NORM_EPS = 1e-6
SAMPLE_RATE = 24000
HOP_LENGTH = 3200  # product of downsampling_ratios [2,2,4,5,5,8]

# Acoustic/semantic encoder dims
VAE_DIM = 64
SEM_DIM = 128

# VAE noise injection std for reparameterization at inference.
# From acoustic_tokenizer_encoder.vae_std in the HF model config.
# Applied as: latents += N(0,1) * vae_std * N(0,1) (per-batch scale, per-element noise).
VAE_STD = 0.625

# Chunked encoding: 60s per chunk
CHUNK_SECONDS = 60
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 1,440,000
CHUNK_TOKENS = CHUNK_SAMPLES // HOP_LENGTH   # 450

# Special token IDs (from Qwen2 tokenizer, repurposed by VibeVoice-ASR)
AUDIO_TOKEN_ID = 151648    # <|box_start|> — placeholder replaced with audio embeddings
AUDIO_BOS_ID = 151646      # <|object_ref_start|> — marks start of audio region
AUDIO_EOS_ID = 151647      # <|object_ref_end|> — marks end of audio region
EOS_ID = 151643            # <|endoftext|> — generation stop token

# RoPE
ROPE_THETA = 1000000.0
MAX_SEQ_LEN = 32768

BUILD_DIR = Path(__file__).resolve().parent.parent / "build/vibevoice-asr"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

DEFAULT_AUDIO = None  # must be provided by user

# ─── Prompt construction ─────────────────────────────────────────────────────

def build_prompt_ids(num_audio_tokens: int, duration_secs: float,
                     prompt: Optional[str] = None) -> List[int]:
    """Build the full input_ids sequence for ASR inference.

    Format:
      <|im_start|>system\nYou are a helpful assistant...<|im_end|>\n
      <|im_start|>user\n<audio_bos><audio_token>*N<audio_eos>\n
      This is a {duration}s audio, please transcribe...<|im_end|>\n
      <|im_start|>assistant\n

    Built manually because apply_chat_template in transformers 5.x
    strips special tokens from message content.
    """
    _ensure_prompt_constants()
    tokenizer = _get_tokenizer()

    if prompt:
        user_text = (
            f"This is a {duration_secs:.2f} seconds audio, with extra info: {prompt}\n"
            f"Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )
    else:
        user_text = (
            f"This is a {duration_secs:.2f} seconds audio, "
            f"please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )

    IM_START = 151644  # <|im_start|>
    IM_END = 151645    # <|im_end|>

    audio_section = [AUDIO_BOS_ID] + [AUDIO_TOKEN_ID] * num_audio_tokens + [AUDIO_EOS_ID]
    user_text_ids = tokenizer.encode(f"\n{user_text}", add_special_tokens=False)

    input_ids = (
        _PROMPT_PREFIX
        + audio_section + user_text_ids + [IM_END, _NL_TOKEN]
        + _PROMPT_SUFFIX
    )
    return input_ids


# Pre-computed token sequences for the fixed parts of the prompt
_tokenizer_cache = None

def _get_tokenizer():
    global _tokenizer_cache
    if _tokenizer_cache is None:
        from transformers import AutoTokenizer
        _tokenizer_cache = AutoTokenizer.from_pretrained(
            "microsoft/VibeVoice-ASR-HF", trust_remote_code=True)
    return _tokenizer_cache

def _init_prompt_constants():
    tok = _get_tokenizer()
    system_msg = "You are a helpful assistant that transcribes audio input into text output in JSON format."
    nl = tok.encode("\n", add_special_tokens=False)[0]
    system_ids = tok.encode(f"system\n{system_msg}", add_special_tokens=False)
    user_prefix = tok.encode("user\n", add_special_tokens=False)
    assistant_ids = tok.encode("assistant\n", add_special_tokens=False)
    IM_START = 151644
    IM_END = 151645
    prefix = [IM_START] + system_ids + [IM_END, nl, IM_START] + user_prefix
    suffix = [IM_START] + assistant_ids
    return prefix, suffix, nl

_PROMPT_PREFIX, _PROMPT_SUFFIX, _NL_TOKEN = None, None, None

def _ensure_prompt_constants():
    global _PROMPT_PREFIX, _PROMPT_SUFFIX, _NL_TOKEN
    if _PROMPT_PREFIX is None:
        _PROMPT_PREFIX, _PROMPT_SUFFIX, _NL_TOKEN = _init_prompt_constants()


# ─── Embedding loading ───────────────────────────────────────────────────────

def load_embeddings(path: Path) -> np.ndarray:
    """Load token embeddings from .bin (float16 with uint32 header) → float32 array."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        vocab_size, hidden_size = int(header[0]), int(header[1])
        data = np.frombuffer(f.read(), dtype=np.float16)
    return data.reshape(vocab_size, hidden_size).astype(np.float32)


# ─── Audio loading ───────────────────────────────────────────────────────────

def load_audio(audio_path: str) -> np.ndarray:
    """Load audio file to mono 24kHz float32, normalize to -25 dB FS."""
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        gcd = math.gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)

    # Normalize to -25 dB FS (RMS-based, matching VibeVoice feature extractor)
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 0:
        target_rms = 10 ** (-25 / 20)
        wav = wav * (target_rms / rms)

    # Pad to multiple of HOP_LENGTH
    remainder = len(wav) % HOP_LENGTH
    if remainder > 0:
        wav = np.pad(wav, (0, HOP_LENGTH - remainder))

    return wav.astype(np.float32)


# ─── RoPE computation ────────────────────────────────────────────────────────

def compute_rope_np(position: int, head_dim: int) -> tuple:
    """Compute RoPE cos/sin for a single position. Returns (1, 1, head_dim)."""
    half_dim = head_dim // 2
    freq_seq = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (ROPE_THETA ** (freq_seq / half_dim))
    angle = position * inv_freq
    cos_val = np.cos(angle).astype(np.float32)
    sin_val = np.sin(angle).astype(np.float32)
    cos_full = np.concatenate([cos_val, cos_val]).reshape(1, 1, head_dim)
    sin_full = np.concatenate([sin_val, sin_val]).reshape(1, 1, head_dim)
    return cos_full, sin_full


def compute_rope_batch_np(start_pos: int, seq_len: int, head_dim: int) -> tuple:
    """Compute RoPE cos/sin for a range of positions. Returns (1, seq_len, head_dim)."""
    half_dim = head_dim // 2
    freq_seq = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (ROPE_THETA ** (freq_seq / half_dim))
    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)  # (seq_len, half_dim)
    cos_val = np.cos(angles).astype(np.float32)
    sin_val = np.sin(angles).astype(np.float32)
    cos_full = np.concatenate([cos_val, cos_val], axis=1).reshape(1, seq_len, head_dim)
    sin_full = np.concatenate([sin_val, sin_val], axis=1).reshape(1, seq_len, head_dim)
    return cos_full, sin_full


# ─── Metrics helper ──────────────────────────────────────────────────────────

class PipelineMetrics:
    def __init__(self, name: str):
        self.name = name
        self.timings = {}
        self.total_time = 0.0
        self.num_audio_tokens = 0
        self.num_prompt_tokens = 0
        self.num_generated_tokens = 0
        self.peak_memory_mb = 0.0

    def record(self, component: str, ms: float):
        if component not in self.timings:
            self.timings[component] = []
        self.timings[component].append(ms)

    def summary(self) -> dict:
        result = {"name": self.name}
        for k, v in self.timings.items():
            result[f"{k}_total_ms"] = sum(v)
            result[f"{k}_mean_ms"] = sum(v) / len(v) if v else 0
            result[f"{k}_count"] = len(v)
        result["total_ms"] = self.total_time
        result["audio_tokens"] = self.num_audio_tokens
        result["prompt_tokens"] = self.num_prompt_tokens
        result["generated_tokens"] = self.num_generated_tokens
        result["peak_memory_mb"] = self.peak_memory_mb
        # Tokens per second
        gen_ms = self.total_time - result.get("load_total_ms", 0) - result.get("encode_total_ms", 0)
        result["gen_ms"] = gen_ms
        if self.num_generated_tokens > 0 and gen_ms > 0:
            result["tokens_per_sec"] = self.num_generated_tokens / (gen_ms / 1000)
        return result


def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
