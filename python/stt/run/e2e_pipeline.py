#!/usr/bin/env python3
"""End-to-end VibeVoice-ASR pipeline: audio → text.

Compares PyTorch (HuggingFace), CoreML, and MLX backends
with identical inputs. Reports accuracy, performance, and system load.

Usage:
    uv run python run/e2e_pipeline.py --audio test.wav
    uv run python run/e2e_pipeline.py --audio test.wav --pytorch --coreml --mlx
    uv run python run/e2e_pipeline.py --audio test.wav --coreml --int8 --fused-lm-head
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
import warnings
from pathlib import Path
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*tokenizer class you load.*")
logging.getLogger("coremltools").setLevel(logging.ERROR)

import pipeline_common as common


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(results: List[dict]):
    print()
    print("=" * 80)
    print("  END-TO-END ASR PIPELINE RESULTS")
    print("=" * 80)

    for r in results:
        name = r["name"]
        print(f"\n  --- {name} ---")
        print(f"    Audio tokens:      {r['audio_tokens']}")
        print(f"    Prompt tokens:     {r['prompt_tokens']}")
        print(f"    Generated tokens:  {r['generated_tokens']}")
        print(f"    Total time:        {r['total_ms']:.0f}ms (load: {r.get('load_total_ms', 0):.0f}ms)")
        if "tokens_per_sec" in r:
            print(f"    Tokens/sec:        {r['tokens_per_sec']:.1f}")
        print(f"    Peak memory delta: {r['peak_memory_mb']:.0f}MB")
        print()
        print(f"    {'Component':<20s} {'Total ms':>10s} {'Mean ms':>10s} {'Count':>8s}")
        print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for comp in ["load", "audio_load", "encode", "prompt_build", "prefill", "lm_step", "generate"]:
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
            ("generated_tokens", "{}"),
            ("gen_ms", "{:.0f}ms"),
            ("tokens_per_sec", "{:.1f}"),
            ("encode_total_ms", "{:.0f}ms"),
            ("prefill_total_ms", "{:.0f}ms"),
            ("lm_step_mean_ms", "{:.2f}ms"),
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
    parser = argparse.ArgumentParser(description="E2E VibeVoice ASR pipeline")
    parser.add_argument("--audio", required=True, help="Input audio file (WAV/MP3/FLAC)")
    parser.add_argument("--prompt", default=None, help="Optional context/prompt for transcription")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument("--pytorch", action="store_true", help="Run PyTorch backend")
    parser.add_argument("--coreml", action="store_true", help="Run CoreML backend")
    parser.add_argument("--mlx", action="store_true", help="Run MLX backend (mlx-audio)")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantized LM (CoreML only)")
    parser.add_argument("--fused-lm-head", action="store_true", help="Use fused LM+head (CoreML only)")
    parser.add_argument("--output-dir", default=None, help="Output directory for transcripts")
    args = parser.parse_args()

    # Default to all backends if none specified
    if not args.pytorch and not args.coreml and not args.mlx:
        args.pytorch = args.coreml = args.mlx = True

    if args.output_dir is None:
        args.output_dir = str(common.OUTPUT_DIR)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    audio_name = Path(args.audio).stem

    print(f"Audio: {args.audio}")
    if args.prompt:
        print(f"Prompt: {args.prompt}")
    print(f"Max new tokens: {args.max_new_tokens}")

    results = []
    transcriptions = {}

    # ─── PyTorch ──────────────────────────────────────────────
    if args.pytorch:
        from pipeline_pytorch import run_pytorch
        print("\n>>> Running PyTorch pipeline...")
        text_pt, metrics_pt = run_pytorch(args.audio, prompt=args.prompt,
                                           max_new_tokens=args.max_new_tokens)
        transcriptions["pytorch"] = text_pt
        results.append(metrics_pt.summary())
        print(f"\n  Transcription preview: {text_pt[:200]}...")

        # Save transcript
        txt_path = out_dir / f"{run_ts}_{audio_name}_pytorch.txt"
        txt_path.write_text(text_pt)
        print(f"  Saved: {txt_path}")
        gc.collect()

    # ─── CoreML ───────────────────────────────────────────────
    if args.coreml:
        from pipeline_coreml import run_coreml
        label = "int8+fused" if args.int8 and args.fused_lm_head else \
                "int8" if args.int8 else \
                "fused" if args.fused_lm_head else "fp16"
        print(f"\n>>> Running CoreML pipeline ({label})...")
        text_cml, metrics_cml = run_coreml(args.audio, prompt=args.prompt,
                                            max_new_tokens=args.max_new_tokens,
                                            int8=args.int8,
                                            fused_lm_head=args.fused_lm_head)
        transcriptions["coreml"] = text_cml
        results.append(metrics_cml.summary())
        print(f"\n  Transcription preview: {text_cml[:200]}...")

        txt_path = out_dir / f"{run_ts}_{audio_name}_coreml_{label}.txt"
        txt_path.write_text(text_cml)
        print(f"  Saved: {txt_path}")
        gc.collect()

    # ─── MLX ─────────────────────────────────────────────────
    if args.mlx:
        from pipeline_mlx import run_mlx
        print("\n>>> Running MLX pipeline...")
        text_mlx, metrics_mlx = run_mlx(args.audio, prompt=args.prompt,
                                         max_new_tokens=args.max_new_tokens)
        transcriptions["mlx"] = text_mlx
        results.append(metrics_mlx.summary())
        print(f"\n  Transcription preview: {text_mlx[:200]}...")

        txt_path = out_dir / f"{run_ts}_{audio_name}_mlx.txt"
        txt_path.write_text(text_mlx)
        print(f"  Saved: {txt_path}")
        gc.collect()

    # ─── Report ───────────────────────────────────────────────
    print_report(results)

    # Compare transcriptions
    if len(transcriptions) > 1:
        print("\n" + "=" * 80)
        print("  TRANSCRIPTION COMPARISON")
        print("=" * 80)
        keys = list(transcriptions.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = transcriptions[keys[i]], transcriptions[keys[j]]
                match = a.strip() == b.strip()
                print(f"\n  {keys[i]} vs {keys[j]}: {'MATCH' if match else 'DIFFER'}")
                if not match:
                    print(f"    {keys[i]}: {a[:100]}...")
                    print(f"    {keys[j]}: {b[:100]}...")


if __name__ == "__main__":
    main()
