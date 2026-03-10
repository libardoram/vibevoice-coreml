"""MLX backend for 0.5B streaming TTS pipeline (via mlx-audio library)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

import pipeline_common as common


def run_mlx(text: str, voice_name: str, cfg_scale: float,
            diffusion_steps: int, seed: int) -> Tuple[np.ndarray, common.Metrics]:
    """Run using mlx-audio library."""
    from mlx_audio.tts.utils import load_model

    metrics = common.Metrics()
    t0 = time.time()

    model = load_model("mlx-community/VibeVoice-Realtime-0.5B-fp16")
    metrics.record("load", time.time() - t0)

    # Map voice name to mlx-audio format (e.g. "Emma" -> "en-Emma_woman")
    # Try local voice files first, fall back to matching from model's voices dir
    try:
        voice_path = common.load_voice_prompt(voice_name)
        mlx_voice = voice_path.stem
    except FileNotFoundError:
        from pathlib import Path
        voices_dir = Path(model.config.model_path) / "voices"
        match = None
        if voices_dir.exists():
            for f in voices_dir.glob("*.safetensors"):
                if voice_name.lower() in f.stem.lower():
                    match = f.stem
                    break
        mlx_voice = match or f"en-{voice_name}_woman"

    t0 = time.time()
    for result in model.generate(
        text=text,
        voice=mlx_voice,
        max_tokens=max(len(text.split()) * 5, 30),
        cfg_scale=cfg_scale,
        ddpm_steps=diffusion_steps,
    ):
        pass  # single result for single-speaker

    gen_time = time.time() - t0
    metrics.record("generate", gen_time)

    audio = np.array(result.audio, dtype=np.float32).flatten()

    return audio, metrics
