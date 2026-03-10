#!/usr/bin/env python3
"""Run all VibeVoice-Realtime-0.5B CoreML conversions in one command.

Runs convert_coreml.py (components) and convert_stateful_lm.py (LM backbone)
sequentially, forwarding shared arguments.

Usage:
    uv run python convert_all.py
    uv run python convert_all.py --output-dir ./build/vibevoice-realtime-0.5b
    uv run python convert_all.py --skip-vae --skip-diffusion
"""

import subprocess
import sys
import time
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

DEFAULT_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "build/vibevoice-realtime-0.5b"


def _run(cmd: list[str], label: str) -> None:
    """Run a conversion subprocess, streaming output."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}\n")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFAILED: {label} (exit {result.returncode}, {elapsed:.0f}s)")
        raise typer.Exit(1)
    print(f"\n  {label} completed in {elapsed:.0f}s")


@app.command()
def convert_all(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    max_seq_len: int = typer.Option(8192, "--max-seq-len"),
    compute_units: str = typer.Option("ALL", "--compute-units"),
    compute_precision: str = typer.Option("FLOAT16", "--compute-precision"),
    skip_vae: bool = typer.Option(False, "--skip-vae"),
    skip_diffusion: bool = typer.Option(False, "--skip-diffusion"),
    skip_lm: bool = typer.Option(False, "--skip-lm", help="Skip stateful LM export"),
) -> None:
    """Run all VibeVoice-Realtime-0.5B CoreML conversions."""
    t_total = time.time()

    # 1. Components (diffusion, VAE, EOS, connector)
    if not (skip_vae and skip_diffusion):
        cmd = [
            "convert_coreml.py",
            "--model-id", model_id,
            "--output-dir", str(output_dir),
            "--compute-units", compute_units,
            "--compute-precision", compute_precision,
        ]
        if skip_vae:
            cmd.append("--skip-vae")
        if skip_diffusion:
            cmd.append("--skip-diffusion")
        _run(cmd, "Components (convert_coreml.py)")
    else:
        # Still need EOS + connector even if VAE/diffusion skipped
        cmd = [
            "convert_coreml.py",
            "--model-id", model_id,
            "--output-dir", str(output_dir),
            "--compute-units", compute_units,
            "--compute-precision", compute_precision,
            "--skip-vae", "--skip-diffusion",
        ]
        _run(cmd, "Components — EOS + connector only (convert_coreml.py)")

    # 2. Stateful LM (base + TTS decoders)
    if not skip_lm:
        cmd = [
            "convert_stateful_lm.py",
            "--model-id", model_id,
            "--output-dir", str(output_dir),
            "--max-seq-len", str(max_seq_len),
        ]
        _run(cmd, "Stateful LM backbone (convert_stateful_lm.py)")

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL CONVERSIONS COMPLETE — {elapsed:.0f}s total")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    app()
