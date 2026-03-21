#!/usr/bin/env python3
"""End-to-end VibeVoice multi-speaker TTS pipeline: text → audio.

Compares PyTorch (fp32/CPU), CoreML (fp16/CPU+GPU), and MLX (fp16/GPU)
with identical inputs. Reports correctness, performance, and system load.

Optimizations (configurable via CLI flags):
  --solver dpm            Use DPM-Solver++ 2M instead of DDPM (fewer steps, same quality)
  --diffusion-steps N     Number of diffusion inference steps (default: 20)
  --int8                  Use INT8 quantized LM decoder (run convert_all.py with --int8/--fuse-lm-head first)
  --fused-lm-head         Use fused LM+head model (run convert_all.py with --int8/--fuse-lm-head first)
  --parallel              Overlap VAE decode with next LM step via threading

Usage:
    uv run python run/e2e_pipeline.py --coreml --solver dpm --diffusion-steps 10 --int8
    uv run python run/e2e_pipeline.py --pytorch --coreml  (run specific backends; omit all for all three)
    uv run python run/e2e_pipeline.py --ref-audio spk1.wav spk2.wav --text $'Speaker 1: Hello\\nSpeaker 2: Hi there'
    uv run python run/e2e_pipeline.py --pytorch --coreml --mlx  (compare all three)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
warnings.filterwarnings("ignore", message=".*tokenizer class you load.*")
logging.getLogger("coremltools").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import torch

import pipeline_common as common


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(results: List[dict]):
    print()
    print("=" * 80)
    print("  END-TO-END PIPELINE RESULTS")
    print("=" * 80)

    for r in results:
        name = r["name"]
        print(f"\n  --- {name} ---")
        print(f"    Text tokens:       {r['text_tokens']}")
        print(f"    Speech tokens:     {r['speech_tokens']}")
        print(f"    Audio:             {r['audio_samples']} samples ({r['audio_seconds']:.2f}s)")
        print(f"    Total time:        {r['total_ms']:.0f}ms (load: {r.get('load_total_ms', 0):.0f}ms)")
        if "gen_rtf" in r:
            print(f"    Generation RTF:    {r['gen_rtf']:.2f}x (excl. model load)")
        if "rtf" in r:
            print(f"    End-to-end RTF:    {r['rtf']:.2f}x (incl. model load)")
        print(f"    Peak memory delta: {r['peak_memory_mb']:.0f}MB")
        print()
        print(f"    {'Component':<20s} {'Total ms':>10s} {'Mean ms':>10s} {'Count':>8s}")
        print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for comp in ["load", "prefill", "lm_step", "diffusion", "vae", "connector"]:
            total_key = f"{comp}_total_ms"
            mean_key = f"{comp}_mean_ms"
            count_key = f"{comp}_count"
            if total_key in r:
                print(f"    {comp:<20s} {r[total_key]:>9.1f}  {r[mean_key]:>9.2f}  {r[count_key]:>7d}")

    # Comparison table
    if len(results) > 1:
        def short_name(name):
            if "(" in name and ")" in name:
                return name[name.index("(") + 1:name.index(")")]
            return name[:18]

        col_w = max(12, max(len(short_name(r["name"])) for r in results) + 2)
        print()
        print("=" * 80)
        print("  COMPARISON")
        print("=" * 80)
        header = f"  {'Metric':<25s}"
        for r in results:
            header += f" {short_name(r['name']):>{col_w}s}"
        print(header)
        print(f"  {'-'*25}" + f" {'-'*col_w}" * len(results))

        for metric, fmt in [
            ("speech_tokens", "{}"),
            ("audio_seconds", "{:.2f}s"),
            ("gen_ms", "{:.0f}ms"),
            ("gen_rtf", "{:.2f}x"),
            ("prefill_total_ms", "{:.0f}ms"),
            ("lm_step_mean_ms", "{:.2f}ms"),
            ("diffusion_mean_ms", "{:.2f}ms"),
            ("vae_mean_ms", "{:.2f}ms"),
            ("connector_mean_ms", "{:.2f}ms"),
            ("load_total_ms", "{:.0f}ms"),
        ]:
            row = f"  {metric:<25s}"
            for r in results:
                val = r.get(metric, "—")
                if val == "—":
                    row += f" {'—':>{col_w}s}"
                else:
                    row += f" {fmt.format(val):>{col_w}s}"
            print(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E VibeVoice TTS pipeline comparison")
    parser.add_argument("--model-id", required=False,
                        choices=list(common.MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--text", default=None, help="Input text (Speaker N: ...)")
    parser.add_argument("--text-file", default=None, help="Read input text from file")
    parser.add_argument("--max-speech-tokens", type=int, default=None,
                        help="Max speech tokens to generate (default: auto from text length)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diffusion noise")
    parser.add_argument("--pytorch", action="store_true", help="Run PyTorch backend")
    parser.add_argument("--coreml", action="store_true", help="Run CoreML backend")
    parser.add_argument("--mlx", action="store_true", help="Run MLX backend")
    parser.add_argument("--output-dir", default=None, help="Output directory for audio (default: auto from model)")
    # Optimization flags
    parser.add_argument("--solver", choices=["ddpm", "dpm"], default="ddpm",
                        help="Diffusion solver: ddpm (default) or dpm (DPM-Solver++ 2M)")
    parser.add_argument("--diffusion-steps", type=int, default=20,
                        help="Number of diffusion inference steps (default: 20)")
    parser.add_argument("--cfg-scale", type=float, default=1.3,
                        help="Classifier-free guidance scale (default: 1.3, 1.0=no guidance)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantized LM decoder (CoreML only)")
    parser.add_argument("--fused-lm-head", action="store_true",
                        help="Use fused LM+head model (CoreML only)")
    parser.add_argument("--fused-diffusion", action="store_true",
                        help="Use fused diffusion loop (CoreML only)")
    parser.add_argument("--parallel", action="store_true",
                        help="Overlap VAE decode with next LM step (CoreML only)")
    parser.add_argument("--lm-compute", choices=["all", "cpu_gpu", "cpu"], default="cpu_gpu",
                        help="Compute units for LM model (default: cpu_gpu)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multiple CoreML configs and compare")
    parser.add_argument("--ref-audio", type=str, nargs="+", default=None,
                        help="Reference audio file(s) for voice cloning, one per speaker (WAV/MP3)")
    args = parser.parse_args()

    if args.text_file:
        args.text = Path(args.text_file).read_text().strip()
    elif args.text is None:
        args.text = common.DEFAULT_TEXT

    if args.model_id is None:
        parser.error("--model-id is required")

    # Default to all backends if none specified
    if not args.pytorch and not args.coreml and not args.mlx and not args.sweep:
        args.pytorch = args.coreml = args.mlx = True

    # Configure model-specific globals
    common.configure(args.model_id)
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent.parent / common.MODEL_CONFIGS[args.model_id]["output_dir"])

    opt = common.OptConfig(
        solver=args.solver,
        diffusion_steps=args.diffusion_steps,
        cfg_scale=args.cfg_scale,
        int8=args.int8,
        fused_lm_head=args.fused_lm_head,
        fused_diffusion=args.fused_diffusion,
        parallel=args.parallel,
        lm_compute=args.lm_compute,
    )

    # Estimate max speech tokens from text if not specified
    auto_tokens = args.max_speech_tokens is None
    if auto_tokens:
        text_only = re.sub(r"Speaker\s+\d+\s*:", "", args.text)
        word_count = len(text_only.split())
        args.max_speech_tokens = max(common.MIN_SPEECH_TOKENS, int(word_count * common.SPEECH_TOKENS_PER_WORD))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for unique output filenames (shared across backends in same run)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"Text: {args.text}")
    token_note = f"auto, {len(re.sub(r'Speaker *[0-9]+:', '', args.text).split())} words" if auto_tokens else "manual"
    print(f"Max speech tokens: {args.max_speech_tokens} ({token_note})")
    print(f"Seed: {args.seed}")
    print(f"Optimizations: solver={opt.solver}, diffusion_steps={opt.diffusion_steps}, "
          f"cfg_scale={opt.cfg_scale}, int8={opt.int8}, fused={opt.fused_lm_head}, "
          f"fused_diff={opt.fused_diffusion}, parallel={opt.parallel}")
    if args.ref_audio:
        for i, path in enumerate(args.ref_audio):
            print(f"Voice cloning Speaker {i + 1}: {path}")

    prompt_result = common.tokenize_prompt(args.text, ref_audio=args.ref_audio)
    if isinstance(prompt_result, common.VoiceCloneData):
        voice_clone = prompt_result
        input_ids = voice_clone.input_ids
        total_voice_tokens = sum(s.num_vae_tokens for s in voice_clone.speakers)
        print(f"Prompt tokens: {len(input_ids)} (incl. {total_voice_tokens} voice tokens for {len(voice_clone.speakers)} speaker(s))")
    else:
        voice_clone = None
        input_ids = prompt_result
        print(f"Prompt tokens: {len(input_ids)}")

    def _save_result(backend: str, audio: np.ndarray, metrics: common.PipelineMetrics, label_suffix: str = ""):
        """Save audio WAV, return metrics summary."""
        fname = f"{run_ts}_{backend}{label_suffix}.wav"
        wav_path = out_dir / fname
        if len(audio) > 0:
            sf.write(str(wav_path), audio, common.SAMPLE_RATE)
            print(f"  Saved: {wav_path} ({len(audio) / common.SAMPLE_RATE:.2f}s)")
        else:
            print("  No audio generated")
        return metrics.summary()

    # ─── Sweep mode: run multiple CoreML configs ──────────────
    if args.sweep:
        from pipeline_coreml import run_coreml
        results = []
        configs = [
            common.OptConfig(solver="ddpm", diffusion_steps=20),
            common.OptConfig(solver="dpm", diffusion_steps=20),
            common.OptConfig(solver="dpm", diffusion_steps=10),
            common.OptConfig(solver="dpm", diffusion_steps=5),
        ]
        if (common.BUILD_DIR / "lm_decoder_stateful_int8.mlpackage").exists():
            configs += [
                common.OptConfig(solver="ddpm", diffusion_steps=20, int8=True),
                common.OptConfig(solver="dpm", diffusion_steps=10, int8=True),
                common.OptConfig(solver="dpm", diffusion_steps=10, int8=True, parallel=True),
            ]
        if (common.BUILD_DIR / "lm_decoder_fused_int8.mlpackage").exists():
            configs.append(
                common.OptConfig(solver="dpm", diffusion_steps=10, int8=True, fused_lm_head=True, parallel=True),
            )

        for i, cfg in enumerate(configs):
            print(f"\n>>> [{i+1}/{len(configs)}] CoreML with {cfg.label}...")
            try:
                audio, metrics = run_coreml(input_ids, args.max_speech_tokens, args.seed, cfg, voice_clone=voice_clone)
                suffix = f"_{cfg.label.replace('+', '_')}"
                results.append(_save_result("coreml", audio, metrics, label_suffix=suffix))
            except Exception as e:
                print(f"  FAILED: {e}")
            gc.collect()

        print_report(results)
        return

    results = []

    # ─── PyTorch ──────────────────────────────────────────────
    if args.pytorch:
        from pipeline_pytorch import run_pytorch
        print("\n>>> Running PyTorch pipeline...")
        audio_pt, metrics_pt = run_pytorch(input_ids, args.max_speech_tokens, args.seed, opt, voice_clone=voice_clone)
        results.append(_save_result("pytorch", audio_pt, metrics_pt))
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── CoreML ───────────────────────────────────────────────
    if args.coreml:
        from pipeline_coreml import run_coreml
        print(f"\n>>> Running CoreML pipeline ({opt.label})...")
        audio_cml, metrics_cml = run_coreml(input_ids, args.max_speech_tokens, args.seed, opt, voice_clone=voice_clone)
        results.append(_save_result("coreml", audio_cml, metrics_cml))
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── MLX ──────────────────────────────────────────────────
    if args.mlx:
        from pipeline_mlx import run_mlx
        print(f"\n>>> Running MLX pipeline ({opt.solver}-{opt.diffusion_steps}s)...")
        audio_mlx, metrics_mlx = run_mlx(input_ids, args.max_speech_tokens, args.seed, opt, voice_clone=voice_clone)
        results.append(_save_result("mlx", audio_mlx, metrics_mlx))

    # ─── Report ───────────────────────────────────────────────
    print_report(results)


if __name__ == "__main__":
    main()
