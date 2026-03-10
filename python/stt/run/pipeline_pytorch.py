"""PyTorch reference backend for ASR pipeline."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

import pipeline_common as common


def run_pytorch(
    audio_path: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 4096,
) -> tuple[str, common.PipelineMetrics]:
    """Full ASR pipeline using HuggingFace PyTorch model.

    Returns (transcription_text, metrics).
    """
    metrics = common.PipelineMetrics("PyTorch (fp32/CPU)")
    mem_before = common.get_peak_memory_mb()
    t0_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained("microsoft/VibeVoice-ASR-HF",
                                               trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "microsoft/VibeVoice-ASR-HF", dtype=torch.float32,
    )
    model.eval()
    metrics.record("load", (time.perf_counter() - t0) * 1000)

    # Load and preprocess audio
    t0 = time.perf_counter()
    import soundfile as sf
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    duration = len(wav) / sr
    metrics.record("audio_load", (time.perf_counter() - t0) * 1000)

    # Process via HF processor
    t0 = time.perf_counter()
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"},
        ]},
    ]
    if prompt:
        conversation[1]["content"][1]["text"] = (
            f"This is a {duration:.2f} seconds audio, with extra info: {prompt}\n"
            f"Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_path],
                       return_tensors="pt", padding=True)

    num_audio_tokens = (inputs.input_ids == common.AUDIO_TOKEN_ID).sum().item()
    metrics.num_audio_tokens = num_audio_tokens
    metrics.num_prompt_tokens = inputs.input_ids.shape[1]
    metrics.record("encode", (time.perf_counter() - t0) * 1000)

    # Generate
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    metrics.record("generate", (time.perf_counter() - t0) * 1000)

    # Decode
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    metrics.num_generated_tokens = len(generated_ids)
    transcription = processor.decode(generated_ids, skip_special_tokens=True)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.peak_memory_mb = common.get_peak_memory_mb() - mem_before

    return transcription, metrics
