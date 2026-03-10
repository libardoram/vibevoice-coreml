#!/usr/bin/env python3
"""End-to-end VibeVoice-Realtime-0.5B streaming TTS pipeline: text -> audio.

Supports three backends: PyTorch (reference), CoreML, and MLX.
Uses pre-computed voice prompts (.pt files) with cached KV states.

The streaming architecture uses windowed text/speech interleaving:
  - 5 text tokens processed through base LM (4L) then TTS LM (20L)
  - 6 speech frames generated per window via diffusion + VAE decode
  - EOS classifier checks after each speech frame
  - Classifier-free guidance with separate negative KV caches

Usage:
    uv run python run/e2e_pipeline.py                          # all backends
    uv run python run/e2e_pipeline.py --coreml                 # CoreML only
    uv run python run/e2e_pipeline.py --pytorch --coreml --mlx # explicit all
    uv run python run/e2e_pipeline.py --coreml --voice Emma --text "Hello world"
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
warnings.filterwarnings("ignore", message=".*tokenizer class you load.*")
import logging
logging.getLogger("coremltools").setLevel(logging.ERROR)

import soundfile as sf

import pipeline_common as common


def main():
    parser = argparse.ArgumentParser(description="VibeVoice-Realtime-0.5B E2E Pipeline")
    parser.add_argument("--text", default=common.DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--voice", default=common.DEFAULT_VOICE, help="Voice preset name")
    parser.add_argument("--voice-dir", type=Path, default=None, help="Voice prompt directory (default: voices/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diffusion noise")
    parser.add_argument("--pytorch", action="store_true", help="Run PyTorch backend")
    parser.add_argument("--coreml", action="store_true", help="Run CoreML backend")
    parser.add_argument("--mlx", action="store_true", help="Run MLX backend")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "output",
                        help="Output directory for audio")
    parser.add_argument("--cfg-scale", type=float, default=common.DEFAULT_CFG_SCALE,
                        help=f"Classifier-free guidance scale (default: {common.DEFAULT_CFG_SCALE})")
    parser.add_argument("--diffusion-steps", type=int, default=common.DEFAULT_INFERENCE_STEPS,
                        help=f"Number of diffusion inference steps (default: {common.DEFAULT_INFERENCE_STEPS})")
    parser.add_argument("--fused-diffusion", action="store_true",
                        help="Use fused diffusion loop (requires diffusion_loop.mlpackage)")
    args = parser.parse_args()

    # Default to all backends if none specified
    if not args.pytorch and not args.coreml and not args.mlx:
        args.pytorch = args.coreml = args.mlx = True

    if args.voice_dir is not None:
        common.VOICE_DIR = args.voice_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VibeVoice-Realtime-0.5B E2E Pipeline")
    print("=" * 60)
    print(f"Text: {args.text[:80]}...")
    print(f"Voice: {args.voice}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Seed: {args.seed}")
    print()

    # Timestamp for unique output filenames (shared across backends in same run)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    backends = []
    if args.pytorch:
        backends.append("pytorch")
    if args.coreml:
        backends.append("coreml")
    if args.mlx:
        backends.append("mlx")

    for backend in backends:
        print(f"\n--- {backend.upper()} ---")
        try:
            if backend == "pytorch":
                from pipeline_pytorch import run_pytorch
                audio, metrics = run_pytorch(
                    args.text, args.voice, args.cfg_scale,
                    args.diffusion_steps, args.seed,
                )
            elif backend == "coreml":
                from pipeline_coreml import run_coreml
                audio, metrics = run_coreml(
                    args.text, args.voice, args.cfg_scale,
                    args.diffusion_steps, args.seed,
                    fused_diffusion=args.fused_diffusion,
                )
            elif backend == "mlx":
                from pipeline_mlx import run_mlx
                audio, metrics = run_mlx(
                    args.text, args.voice, args.cfg_scale,
                    args.diffusion_steps, args.seed,
                )

            audio_secs = len(audio) / common.SAMPLE_RATE
            metrics.summary(audio_secs)

            # Save audio
            out_path = args.output_dir / f"{run_ts}_{backend}.wav"
            sf.write(str(out_path), audio, common.SAMPLE_RATE)
            print(f"  Saved: {out_path}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
