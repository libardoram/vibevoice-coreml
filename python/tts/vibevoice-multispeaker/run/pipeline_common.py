"""Shared constants, configs, metrics, tokenizer, and voice cloning for e2e pipeline."""

from __future__ import annotations

import math
import os
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import numpy as np
import soundfile as sf
import torch

from diffusion import VAE_DIM, make_cfg_fn, sample_latent
from rope import ROPE_THETA, compute_rope_np

# ─── Architecture constants ──────────────────────────────────────────────────

HIDDEN_SIZE = 1536
SEMANTIC_DIM = 128
HEAD_DIM = 128
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
NUM_LAYERS = 28
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6
SAMPLE_RATE = 24000

# Special token IDs
SPEECH_START_ID = 151652
SPEECH_END_ID = 151653
SPEECH_DIFFUSION_ID = 151654
EOS_ID = 151643

# Learned affine transform for speech latents: features = (latents + bias) * scaling
# From model.speech_scaling_factor and model.speech_bias_factor in the checkpoint.
# These map VAE latent space to the LM's expected input distribution.
# Note: 1.5B/7B use the inverse direction vs 0.5B (multiply vs divide).
SPEECH_SCALING_FACTOR = 0.1962890625
SPEECH_BIAS_FACTOR = -0.04931640625

BUILD_DIR = None  # set by configure()

DEFAULT_TEXT = "Speaker 1: Hello, welcome to the VibeVoice text to speech benchmark."

# Speech token estimation: ~150 wpm, 7.5 tokens/sec → ~3 tokens/word, 1.5x safety margin
SPEECH_TOKENS_PER_WORD = 4.5
MIN_SPEECH_TOKENS = 20

# ─── Multi-model configuration ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "microsoft/VibeVoice-1.5B": {
        "hidden_size": 1536, "num_q_heads": 12, "num_kv_heads": 2,
        "head_dim": 128, "num_layers": 28, "vocab_size": 151936,
        "build_dir": "build/vibevoice-1.5b",
        "tokenizer": "Qwen/Qwen2.5-1.5B",
        "output_dir": "run/output/vibevoice-1.5b",
    },
    "vibevoice/VibeVoice-7B": {
        "hidden_size": 3584, "num_q_heads": 28, "num_kv_heads": 4,
        "head_dim": 128, "num_layers": 28, "vocab_size": 152064,
        "build_dir": "build/vibevoice-7b",
        "tokenizer": "Qwen/Qwen2.5-7B",
        "output_dir": "run/output/vibevoice-7b",
    },
}

MODEL_ID = None       # set by configure()
TOKENIZER_NAME = None  # set by configure()


def configure(model_id):
    """Set module globals from MODEL_CONFIGS for the selected model."""
    global HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, GQA_REPEAT
    global NUM_LAYERS, VOCAB_SIZE, BUILD_DIR, MODEL_ID, TOKENIZER_NAME
    cfg = MODEL_CONFIGS[model_id]
    HIDDEN_SIZE = cfg["hidden_size"]
    NUM_Q_HEADS = cfg["num_q_heads"]
    NUM_KV_HEADS = cfg["num_kv_heads"]
    GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
    NUM_LAYERS = cfg["num_layers"]
    VOCAB_SIZE = cfg["vocab_size"]
    BUILD_DIR = Path(__file__).resolve().parent.parent / cfg["build_dir"]
    MODEL_ID = model_id
    TOKENIZER_NAME = cfg["tokenizer"]


# ─── Optimization config ─────────────────────────────────────────────────────

@dataclass
class OptConfig:
    solver: str = "ddpm"          # "ddpm" or "dpm"
    diffusion_steps: int = 20     # number of inference steps
    cfg_scale: float = 1.3        # classifier-free guidance scale (1.0 = no guidance)
    int8: bool = False            # use INT8 quantized LM decoder
    fused_lm_head: bool = False   # use fused LM+head model
    fused_diffusion: bool = False # use fused diffusion loop (single CoreML call)
    parallel: bool = False        # overlap VAE with next LM step
    lm_compute: str = "cpu_gpu"    # LM compute units: "all", "cpu_gpu", "cpu"

    @property
    def label(self) -> str:
        parts = []
        if self.int8:
            parts.append("int8")
        if self.fused_lm_head:
            parts.append("fused")
        if self.fused_diffusion:
            parts.append("fuseddiff")
        parts.append(f"{self.solver}-{self.diffusion_steps}s")
        if self.parallel:
            parts.append("parallel")
        if self.lm_compute != "all":
            parts.append(f"lm-{self.lm_compute}")
        return "+".join(parts) if parts else "baseline"


# ─── Diffusion dispatch ──────────────────────────────────────────────────────

def _sample_latent(diffusion_fn, condition, opt: OptConfig, seed=None):
    """Dispatch to configured diffusion solver."""
    return sample_latent(diffusion_fn, condition, solver=opt.solver,
                         num_steps=opt.diffusion_steps, seed=seed)


# ─── Streaming semantic encoder ──────────────────────────────────────────────

def _encode_semantic_streaming(audio_chunk_np, sem_tok, sem_conn, sem_cache):
    """Encode one audio chunk (3200 samples) via streaming semantic encoder.

    Uses PyTorch with streaming cache — O(1) per step instead of O(n).
    Returns CPU torch tensor [1, 1, 1536].
    """
    device = next(sem_tok.parameters()).device
    audio_t = torch.from_numpy(audio_chunk_np.reshape(1, 1, -1)).float().to(device)
    with torch.no_grad():
        features = sem_tok.encode(
            audio_t, cache=sem_cache,
            sample_indices=torch.tensor([0], device=device),
            use_cache=True,
        ).mean[:, -1:, :]  # [1, 1, 128]
        return sem_conn(features).cpu()  # [1, 1, 1536] on CPU


# ─── Tokenizer / Voice Cloning ───────────────────────────────────────────────

VOICE_CLONE_SAMPLE_SECS = 10  # VAE encoder is exported for fixed 10s input
VOICE_CLONE_SAMPLES = SAMPLE_RATE * VOICE_CLONE_SAMPLE_SECS  # 240000
SPEECH_TOK_COMPRESS_RATIO = 3200  # samples per VAE token


@dataclass
class SpeakerRef:
    """Per-speaker voice reference data."""
    speaker_id: int
    ref_audio_np: np.ndarray  # resampled mono 24kHz audio, shape (N,)
    num_vae_tokens: int
    speech_embed_positions: List[int]  # indices into input_ids where speech embeds go
    _cached_embeds_coreml: Optional[np.ndarray] = None
    _cached_embeds_pt: Optional[np.ndarray] = None


@dataclass
class VoiceCloneData:
    """Pre-processed voice cloning data for prompt injection (multi-speaker)."""
    input_ids: List[int]
    speakers: List[SpeakerRef]


def _load_and_resample(audio_path: str) -> np.ndarray:
    """Load audio file, convert to mono 24kHz float32."""
    from scipy.signal import resample_poly
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        gcd = math.gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
    return wav.astype(np.float32)


def encode_voice_reference_pt(wav: np.ndarray, model, num_vae_tokens: int) -> np.ndarray:
    """Encode reference audio to speech embeddings using PyTorch model.

    Returns embeddings of shape (num_vae_tokens, hidden_size).
    """
    acoustic_tokenizer = model.model.acoustic_tokenizer
    acoustic_connector = model.model.acoustic_connector

    # Pad/trim to full length for VAE encoder
    audio_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    with torch.no_grad():
        encoder_output = acoustic_tokenizer.encode(audio_tensor)
        latents = encoder_output.sample(dist_type=acoustic_tokenizer.std_dist_type)[0]
        # Trim to actual token count
        latents = latents[:, :num_vae_tokens, :]
        # Apply scaling and bias
        features = (latents + model.model.speech_bias_factor) * model.model.speech_scaling_factor
        # Connect to LM space: features is (1, T, vae_dim) → (1, T, hidden_size)
        embeds = acoustic_connector(features)
    return embeds[0].numpy()  # (T, hidden_size)


def encode_voice_reference_coreml(
    wav: np.ndarray,
    num_vae_tokens: int,
    cml_vae_enc=None,
    cml_ac_conn=None,
) -> np.ndarray:
    """Encode reference audio to speech embeddings using CoreML models.

    Pass pre-loaded cml_vae_enc and cml_ac_conn to avoid reloading per call.
    Returns embeddings of shape (num_vae_tokens, hidden_size).
    """
    import coremltools as ct
    if cml_vae_enc is None:
        cml_vae_enc = ct.models.MLModel(str(BUILD_DIR / "vae_encoder.mlpackage"))
    if cml_ac_conn is None:
        cml_ac_conn = ct.models.MLModel(str(BUILD_DIR / "acoustic_connector.mlpackage"))

    # VAE encoder expects fixed 10s = 240000 samples, shape (1, 1, 240000)
    audio_input = np.zeros((1, 1, VOICE_CLONE_SAMPLES), dtype=np.float32)
    actual_len = min(len(wav), VOICE_CLONE_SAMPLES)
    audio_input[0, 0, :actual_len] = wav[:actual_len]

    latents = cml_vae_enc.predict({"audio": audio_input})["latent"]
    # latents shape: (1, vae_dim, T_full) — trim to actual token count
    actual_t = min(latents.shape[2], num_vae_tokens)
    latents = latents[:, :, :actual_t]  # (1, vae_dim, T)

    # Apply scaling and bias, then connect
    features = (latents + SPEECH_BIAS_FACTOR) * SPEECH_SCALING_FACTOR
    # Connector expects (1, 1, vae_dim), process frame by frame
    embeds = []
    for t in range(actual_t):
        frame = features[:, :, t:t+1].transpose(0, 2, 1)  # (1, 1, vae_dim)
        emb = cml_ac_conn.predict({"speech_latent": frame})["embedding"]  # (1, 1, hidden_size)
        embeds.append(emb[0, 0])
    return np.stack(embeds)  # (T, hidden_size)


def tokenize_prompt(text: str, ref_audio: Optional[List[str]] = None) -> "List[int] | VoiceCloneData":
    """Build the full prompt token sequence for TTS.

    If ref_audio is provided (list of audio paths, one per speaker), returns
    VoiceCloneData with voice prompt tokens and per-speaker embedding positions.
    Speaker N in text (1-based) maps to ref_audio[N-1].
    """
    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(TOKENIZER_NAME)

    system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    system_tokens = tokenizer.encode(system_prompt)

    if ref_audio is not None and len(ref_audio) > 0:
        # Voice cloning: add voice input section with per-speaker references
        voice_prefix = tokenizer.encode(" Voice input:\n", add_special_tokens=False)
        newline_tok = tokenizer.encode("\n", add_special_tokens=False)

        voice_tokens = list(voice_prefix)
        speakers = []
        current_offset = len(system_tokens) + len(voice_prefix)

        for spk_idx, audio_path in enumerate(ref_audio):
            wav = _load_and_resample(audio_path)
            if len(wav) > VOICE_CLONE_SAMPLES:
                wav = wav[:VOICE_CLONE_SAMPLES]
            num_vae_tokens = math.ceil(len(wav) / SPEECH_TOK_COMPRESS_RATIO)

            spk_prefix = tokenizer.encode(f" Speaker {spk_idx}:", add_special_tokens=False)
            voice_tokens += spk_prefix
            current_offset += len(spk_prefix)

            voice_tokens.append(SPEECH_START_ID)
            current_offset += 1  # speech_start

            speech_positions = list(range(current_offset, current_offset + num_vae_tokens))
            voice_tokens += [SPEECH_DIFFUSION_ID] * num_vae_tokens
            current_offset += num_vae_tokens

            voice_tokens.append(SPEECH_END_ID)
            voice_tokens += newline_tok
            current_offset += 1 + len(newline_tok)  # speech_end + newline

            speakers.append(SpeakerRef(
                speaker_id=spk_idx,
                ref_audio_np=wav,
                num_vae_tokens=num_vae_tokens,
                speech_embed_positions=speech_positions,
            ))

        # Text section — normalize speaker IDs from 1-based (user) to 0-based (model)
        text_section = tokenizer.encode(" Text input:\n", add_special_tokens=False)
        lines = text.strip().split("\n")
        speaker_tokens_list = []
        for line in lines:
            def _decrement_speaker(m):
                n = int(m.group(1))
                return f"Speaker {max(0, n - 1)}"
            line_clean = re.sub(r"Speaker\s+(\d+)", _decrement_speaker, line.strip())
            speaker_tokens_list += tokenizer.encode(f" {line_clean}\n", add_special_tokens=False)

        output_section = tokenizer.encode(" Speech output:\n", add_special_tokens=False)
        all_ids = (system_tokens + voice_tokens + text_section +
                   speaker_tokens_list + output_section + [SPEECH_START_ID])

        return VoiceCloneData(
            input_ids=all_ids,
            speakers=speakers,
        )

    text_section = tokenizer.encode(" Text input:\n", add_special_tokens=False)

    # Parse "Speaker N: text" lines
    lines = text.strip().split("\n")
    speaker_tokens = []
    for line in lines:
        speaker_tokens += tokenizer.encode(f" {line.strip()}\n", add_special_tokens=False)

    output_section = tokenizer.encode(" Speech output:\n", add_special_tokens=False)

    return system_tokens + text_section + speaker_tokens + output_section + [SPEECH_START_ID]


# ─── Metrics helper ──────────────────────────────────────────────────────────

class PipelineMetrics:
    def __init__(self, name: str):
        self.name = name
        self.timings = {}
        self.total_time = 0.0
        self.num_speech_tokens = 0
        self.num_text_tokens = 0
        self.audio_samples = 0
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
        result["speech_tokens"] = self.num_speech_tokens
        result["text_tokens"] = self.num_text_tokens
        result["audio_samples"] = self.audio_samples
        result["audio_seconds"] = self.audio_samples / SAMPLE_RATE
        result["peak_memory_mb"] = self.peak_memory_mb
        if self.audio_samples > 0 and self.total_time > 0:
            audio_ms = self.audio_samples / SAMPLE_RATE * 1000
            result["rtf"] = audio_ms / self.total_time
        # Generation RTF (excluding model load)
        load_ms = result.get("load_total_ms", 0)
        gen_ms = self.total_time - load_ms
        result["gen_ms"] = gen_ms
        if self.audio_samples > 0 and gen_ms > 0:
            audio_ms = self.audio_samples / SAMPLE_RATE * 1000
            result["gen_rtf"] = audio_ms / gen_ms
        return result


def load_embeddings(path) -> np.ndarray:
    """Load token embeddings from .bin (float16 with uint32 header) → float32 array."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        vocab_size, hidden_size = int(header[0]), int(header[1])
        data = np.frombuffer(f.read(), dtype=np.float16)
    return data.reshape(vocab_size, hidden_size).astype(np.float32)


def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


