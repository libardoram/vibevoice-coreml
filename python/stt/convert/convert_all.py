#!/usr/bin/env python3
"""Convert and optimize VibeVoice-ASR-HF to CoreML.

Runs all conversion steps sequentially:
  1. Components (encoders, projectors, lm_head, embed_tokens)
  2. Stateful LM decoder (Qwen2-7B with KV cache)
  3. Optimizations: INT8 quantization, fused LM+head (optional)

Usage:
    uv run python convert_all.py
    uv run python convert_all.py --int8 --fuse-lm-head
    uv run python convert_all.py --skip-encoders --skip-lm
"""

import glob
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"

# Qwen2-7B config (same as TTS 7B)
CFG = {
    "num_layers": 28,
    "num_q_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "default_max_seq": 32768,
}


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


# ─── INT8 Weight Quantization ───────────────────────────────────────────────

def quantize_weights(input_path: Path, output_path: Path):
    """Apply INT8 symmetric weight quantization (W8A16) to a CoreML model."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    print(f"  Loading {input_path.name}...")
    model = ct.models.MLModel(str(input_path))

    print("  Quantizing weights to INT8 (linear_symmetric, per_block)...")
    t0 = time.time()
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8",
        granularity="per_block", block_size=32,
    )
    config = OptimizationConfig(global_config=op_config)
    quantized = linear_quantize_weights(model, config)
    print(f"  Quantized in {time.time() - t0:.1f}s")

    quantized.save(str(output_path))
    print(f"  Saved: {output_path}")

    orig_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
    quant_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    ratio = quant_size / orig_size if orig_size > 0 else 0
    print(f"  Size: {orig_size / 1e6:.1f}MB -> {quant_size / 1e6:.1f}MB ({ratio:.1%})")


# ─── Fused LM Decoder + Head ────────────────────────────────────────────────

def create_fused_lm_head(output_path: Path, int8: bool = False):
    """Create a fused LM decoder + LM head CoreML model."""
    import coremltools as ct
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    from convert_stateful_lm import StatefulQwen2Decoder

    cfg = CFG
    MAX_SEQ_LEN = cfg["default_max_seq"]

    print(f"  Loading weights for fused model ({MODEL_ID})...")
    t0 = time.time()

    text_config = Qwen2Config(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_q_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        hidden_act="silu",
    )
    text_model = Qwen2Model(text_config)
    text_model.eval()

    model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])
    safetensor_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))
    all_weights = {}
    for f in safetensor_files:
        all_weights.update(load_file(f))

    # ASR uses "language_model.model." prefix
    decoder_weights = {}
    for k, v in all_weights.items():
        if k.startswith("language_model.model."):
            decoder_weights[k[len("language_model.model."):]] = v.float()

    text_model.load_state_dict(decoder_weights, strict=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Use actual lm_head.weight if available (untied weights)
    if "language_model.lm_head.weight" in all_weights:
        lm_head_weight = all_weights["language_model.lm_head.weight"].float()
        print("  Using separate lm_head.weight (untied)")
    else:
        lm_head_weight = text_model.embed_tokens.weight.detach().float()
        print("  Using embed_tokens as lm_head (tied)")

    class FusedDecoderWithHead(nn.Module):
        def __init__(self, base_decoder, head_weight):
            super().__init__()
            self.decoder = base_decoder
            self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
            self.head.weight.data = head_weight

        def forward(self, hidden_states, position_cos, position_sin, attention_mask):
            h = self.decoder(hidden_states, position_cos, position_sin, attention_mask)
            logits = self.head(h)
            return h, logits

    base_decoder = StatefulQwen2Decoder(
        text_model.layers, text_model.norm, cfg, MAX_SEQ_LEN
    )
    fused = FusedDecoderWithHead(base_decoder, lm_head_weight)
    fused.eval()

    print("  Tracing fused model...")
    t0 = time.time()
    trace_h = torch.randn(1, 1, cfg["hidden_size"])
    trace_cos = torch.randn(1, 1, cfg["head_dim"])
    trace_sin = torch.randn(1, 1, cfg["head_dim"])
    trace_mask = torch.zeros(1, 1, 1, 5)

    with torch.no_grad():
        traced = torch.jit.trace(fused, (trace_h, trace_cos, trace_sin, trace_mask))
    traced.eval()
    print(f"  Traced in {time.time() - t0:.1f}s")

    del all_weights, decoder_weights, text_model, fused, base_decoder
    import gc
    gc.collect()
    print("  Freed PyTorch memory")

    print("  Converting to CoreML...")
    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, cfg["hidden_size"]), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, cfg["head_dim"]), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, cfg["head_dim"]), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType("output_hidden", dtype=np.float32),
        ct.TensorType("logits", dtype=np.float32),
    ]

    # State names use "decoder." prefix because of FusedDecoderWithHead wrapper
    states = []
    for i in range(cfg["num_layers"]):
        for prefix in ("k", "v"):
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, cfg["num_kv_heads"], MAX_SEQ_LEN, cfg["head_dim"]),
                    dtype=np.float16,
                ),
                name=f"decoder.{prefix}_cache_{i}",
            ))

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print(f"  CoreML conversion in {time.time() - t0:.1f}s")

    mlmodel.save(str(output_path))
    print(f"  Saved: {output_path}")

    # Validate
    print("  Validating...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, cfg["hidden_size"]).astype(np.float32),
            "position_cos": np.random.randn(1, 1, cfg["head_dim"]).astype(np.float32),
            "position_sin": np.random.randn(1, 1, cfg["head_dim"]).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        out = mlmodel.predict(test_input, state=state)
        print(f"  output_hidden: {out['output_hidden'].shape}")
        print(f"  logits: {out['logits'].shape}")
        print(f"  OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    del mlmodel

    if int8:
        quant_path = output_path.parent / output_path.name.replace(
            ".mlpackage", "_int8.mlpackage"
        )
        print("\n  Quantizing fused model to INT8...")
        quantize_weights(output_path, quant_path)


# ─── Main ────────────────────────────────────────────────────────────────────

@app.command()
def convert_all(
    output_dir: Path = typer.Option(None, "--output-dir", help="Output directory"),
    compute_units: str = typer.Option("ALL", "--compute-units"),
    compute_precision: str = typer.Option("FLOAT16", "--compute-precision"),
    skip_encoders: bool = typer.Option(False, "--skip-encoders"),
    skip_projectors: bool = typer.Option(False, "--skip-projectors"),
    skip_lm: bool = typer.Option(False, "--skip-lm", help="Skip stateful LM export"),
    fuse_encoders: bool = typer.Option(False, "--fuse-encoders", help="Export fused acoustic+semantic encoder"),
    fuse_projectors: bool = typer.Option(False, "--fuse-projectors", help="Export fused acoustic+semantic projector"),
    int8: bool = typer.Option(False, "--int8", help="INT8 quantize the LM decoder (W8A16)"),
    fuse_lm_head: bool = typer.Option(False, "--fuse-lm-head", help="Create fused LM+head model"),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Keep all intermediate models"),
    compile: bool = typer.Option(False, "--compile", help="Compile .mlpackage to .mlmodelc"),
) -> None:
    """Convert and optimize VibeVoice-ASR-HF to CoreML."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "build/vibevoice-asr"
    t_total = time.time()

    # 1. Components (encoders, projectors, lm_head, embed_tokens)
    cmd = [
        "convert_coreml.py",
        "--output-dir", str(output_dir),
        "--compute-units", compute_units,
        "--compute-precision", compute_precision,
    ]
    if skip_encoders:
        cmd.append("--skip-encoders")
    if skip_projectors:
        cmd.append("--skip-projectors")
    if fuse_encoders:
        cmd.append("--fuse-encoders")
    if fuse_projectors:
        cmd.append("--fuse-projectors")
    _run(cmd, "Components (convert_coreml.py)")

    # 2. Stateful LM decoder
    if not skip_lm:
        _run(
            ["convert_stateful_lm.py", "--output-dir", str(output_dir)],
            "Stateful LM backbone (convert_stateful_lm.py)",
        )

    # 3. Optimizations
    if int8 and not fuse_lm_head:
        print(f"\n{'=' * 70}")
        print("  INT8 Weight Quantization (LM Decoder)")
        print(f"{'=' * 70}\n")
        orig = output_dir / "lm_decoder_stateful.mlpackage"
        if orig.exists():
            quant_path = output_dir / "lm_decoder_stateful_int8.mlpackage"
            quantize_weights(orig, quant_path)
        else:
            print(f"  WARNING: {orig} not found, skipping INT8 quantization")

    if fuse_lm_head:
        print(f"\n{'=' * 70}")
        print("  Fused LM Decoder + Head")
        print(f"{'=' * 70}\n")
        fused_path = output_dir / "lm_decoder_fused.mlpackage"
        create_fused_lm_head(fused_path, int8=int8)

    # 4. Clean up intermediate models
    cleanup = []
    if int8 and not fuse_lm_head:
        cleanup.append("lm_decoder_stateful.mlpackage")
    if fuse_lm_head and not int8:
        cleanup += ["lm_decoder_stateful.mlpackage", "lm_head.mlpackage"]
    if fuse_lm_head and int8:
        cleanup += [
            "lm_decoder_stateful.mlpackage",
            "lm_decoder_fused.mlpackage",
            "lm_head.mlpackage",
        ]

    if cleanup and not no_cleanup:
        print("\n  Cleaning up intermediate models...")
        for name in cleanup:
            p = output_dir / name
            if p.exists():
                shutil.rmtree(p)
                print(f"    Removed {name}")

    # 5. Compile .mlpackage → .mlmodelc
    if compile:
        import coremltools as ct

        print(f"\n{'=' * 70}")
        print("  Compiling .mlpackage → .mlmodelc")
        print(f"{'=' * 70}\n")
        for pkg in sorted(output_dir.glob("*.mlpackage")):
            dst = output_dir / pkg.name.replace(".mlpackage", ".mlmodelc")
            print(f"  {pkg.name}...", end=" ", flush=True)
            t0 = time.time()
            ct.models.utils.compile_model(str(pkg), str(dst))
            size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
            print(f"→ {size / 1e6:.1f}MB ({time.time() - t0:.1f}s)")

    # Summary
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL CONVERSIONS COMPLETE — {elapsed:.0f}s total")
    print(f"  Output: {output_dir}")
    lm_models = sorted(output_dir.glob("lm_decoder*.mlpackage"))
    if lm_models:
        print()
        for p in lm_models:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"    {p.name}: {size / 1e6:.1f}MB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    app()
