"""PyTorch backend for 0.5B streaming TTS pipeline."""

from __future__ import annotations

import copy
import time
from typing import Tuple

import numpy as np

import pipeline_common as common


def run_pytorch(text: str, voice_name: str, cfg_scale: float,
                diffusion_steps: int, seed: int) -> Tuple[np.ndarray, common.Metrics]:
    """Run using reference PyTorch implementation."""
    import torch
    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    metrics = common.Metrics()

    t0 = time.time()
    processor = VibeVoiceStreamingProcessor.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to(device)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=diffusion_steps)

    voice_path = common.load_voice_prompt(voice_name)
    all_prefilled_outputs = torch.load(str(voice_path), map_location=device, weights_only=False)
    metrics.record("load", time.time() - t0)

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        show_progress_bar=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
    )
    metrics.record("generate", time.time() - t0)

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0].cpu().numpy().flatten()
    else:
        audio = np.zeros(common.SAMPLE_RATE, dtype=np.float32)

    return audio, metrics
