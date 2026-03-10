"""MLX backend for ASR pipeline (via mlx-audio library)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

import pipeline_common as common


def run_mlx(
    audio_path: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 8192,
) -> tuple[str, common.PipelineMetrics]:
    """Full ASR pipeline using mlx-audio.

    Returns (transcription_text, metrics).
    """
    import mlx.core as mx

    metrics = common.PipelineMetrics("MLX (8bit/GPU)")
    mem_before = common.get_peak_memory_mb()
    t0_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    from mlx_audio.stt import load

    model = load("mlx-community/VibeVoice-ASR-8bit")
    mx.eval(model.parameters())
    metrics.record("load", (time.perf_counter() - t0) * 1000)

    # Generate
    t0 = time.perf_counter()
    result = model.generate(
        audio_path,
        context=prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        verbose=True,
    )
    metrics.record("generate", (time.perf_counter() - t0) * 1000)

    metrics.num_prompt_tokens = result.prompt_tokens
    metrics.num_generated_tokens = result.generation_tokens

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    return result.text, metrics
