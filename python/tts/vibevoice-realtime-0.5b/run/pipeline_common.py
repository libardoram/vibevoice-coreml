"""Shared constants, configs, metrics, and utilities for 0.5B streaming TTS pipeline."""

from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from diffusion import VAE_DIM, dpm_solver_2m_sample, make_batched_cfg_fn, make_cfg_fn
from rope import ROPE_THETA, compute_rope_np, compute_rope_np_multi

# ─── Architecture constants ──────────────────────────────────────────────────

HIDDEN_SIZE = 896
HEAD_DIM = 64
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
BASE_LM_LAYERS = 4
TTS_LM_LAYERS = 20
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6
SAMPLE_RATE = 24000

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6

DEFAULT_INFERENCE_STEPS = 5
DEFAULT_CFG_SCALE = 1.5

# Learned affine transform for speech latents: latent = raw / scaling - bias
# From model.speech_scaling_factor and model.speech_bias_factor in the checkpoint.
# These map VAE latent space to the LM's expected input distribution.
SPEECH_SCALING_FACTOR = 0.23339844
SPEECH_BIAS_FACTOR = -0.0703125

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = _PROJECT_ROOT / "build/vibevoice-realtime-0.5b"
VOICE_DIR = _PROJECT_ROOT / "voices"

DEFAULT_TEXT = "Hello, welcome to the VibeVoice real time text to speech system. This is a streaming model that generates speech in small windows."
DEFAULT_VOICE = "Emma"


# ─── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    timings: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def record(self, name, elapsed):
        self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def count(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + n

    def summary(self, audio_secs):
        total_gen = sum(v for k, v in self.timings.items() if k != "load")
        rtf = audio_secs / total_gen if total_gen > 0 else 0
        print(f"\n  Audio: {audio_secs:.2f}s")
        print(f"  Generation: {total_gen*1000:.0f}ms (RTF: {rtf:.2f}x)")
        for k, v in sorted(self.timings.items()):
            suffix = f" ({self.counts.get(k, '?')}x)" if k in self.counts else ""
            print(f"  {k}: {v*1000:.1f}ms{suffix}")


# ─── Voice prompt loading ────────────────────────────────────────────────────

def load_voice_prompt(voice_name: str) -> Path:
    """Load pre-computed voice prompt .pt file."""
    candidates = list(VOICE_DIR.glob("*.pt"))
    for c in candidates:
        name = c.stem.split("-")[-1].split("_")[0]
        if name.lower() == voice_name.lower():
            return c

    if candidates:
        print(f"  Warning: voice '{voice_name}' not found, using {candidates[0].stem}")
        return candidates[0]

    raise FileNotFoundError(f"No voice prompts found in {VOICE_DIR}")


# ─── Tokenizer ───────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> List[int]:
    """Tokenize text for TTS generation."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained("Qwen/Qwen2.5-0.5B")
    return tokenizer.encode(text.strip() + "\n", add_special_tokens=False)


def load_embeddings(path) -> np.ndarray:
    """Load token embeddings from .bin (float16 with uint32 header) → float32 array."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        vocab_size, hidden_size = int(header[0]), int(header[1])
        data = np.frombuffer(f.read(), dtype=np.float16)
    return data.reshape(vocab_size, hidden_size).astype(np.float32)


def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
