#!/usr/bin/env python3
"""Export VibeVoice semantic encoder as a stateful streaming CoreML model.

The σ-VAE encoder has 34 causal SConv1d layers, each needing a small cache
buffer for streaming. This script wraps the encoder so each conv cache is a
CoreML state tensor, enabling O(1) per-step inference (3200 samples → 1 frame)
instead of re-encoding up to 240000 padded samples every step.

Output:
    build/<model>/semantic_encoder_streaming.mlpackage

Usage:
    uv run python convert_streaming_semantic.py
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BUILD_DIR = Path(__file__).resolve().parent.parent / "build/vibevoice-1.5b"  # semantic encoder is shared across models
SAMPLE_RATE = 24000
FRAME_SAMPLES = 3200  # one semantic frame = 3200 audio samples at 24kHz


# ─── Architecture constants ──────────────────────────────────────────────────

N_FILTERS = 32
RATIOS = [2, 2, 4, 5, 5, 8]  # encoder ratios (reversed from decoder)
DEPTHS = [3, 3, 3, 3, 3, 3, 8]
KERNEL_SIZE = 7
LAST_KERNEL_SIZE = 7
OUTPUT_DIM = 128  # semantic tokenizer dimension
RMS_EPS = 1e-5


def _context_size(kernel_size, stride=1, dilation=1):
    return (kernel_size - 1) * dilation - (stride - 1)


# ─── Stateful streaming encoder ──────────────────────────────────────────────

class StatefulStreamingSemanticEncoder(nn.Module):
    """Semantic encoder with explicit conv cache buffers for CoreML state.

    Each SConv1d layer's causal padding context is stored as a registered buffer.
    On each forward call, buffers are read, prepended to the input, and updated
    with the new context — identical to the streaming cache approach but with
    fixed buffer names for CoreML StateType export.
    """

    def __init__(self):
        super().__init__()
        self._conv_layers = []  # (name, weight, bias, stride, groups, context_size, channels_in)
        self._block_params = []  # for Block1D layers
        self._build_architecture()

    def _register_conv(self, name, in_ch, out_ch, kernel_size, stride=1, groups=1):
        """Register a conv layer's weight/bias and its cache buffer."""
        ctx = _context_size(kernel_size, stride)
        # Placeholder weight and bias — will be loaded from pretrained
        self.register_parameter(f"{name}_weight", nn.Parameter(torch.zeros(out_ch, in_ch // groups, kernel_size)))
        self.register_parameter(f"{name}_bias", nn.Parameter(torch.zeros(out_ch)))
        # Cache buffer for streaming state
        self.register_buffer(f"{name}_cache", torch.zeros(1, in_ch, ctx))
        self._conv_layers.append((name, stride, groups, ctx, in_ch))

    def _register_block(self, stage_idx, block_idx, dim):
        """Register a Block1D's parameters: norm, depthwise conv, gamma, ffn_norm, ffn, ffn_gamma."""
        prefix = f"stage{stage_idx}_block{block_idx}"
        # Mixer conv (depthwise, groups=dim)
        self._register_conv(f"{prefix}_conv", dim, dim, KERNEL_SIZE, stride=1, groups=dim)
        # Norms
        self.register_parameter(f"{prefix}_norm_weight", nn.Parameter(torch.ones(dim)))
        self.register_parameter(f"{prefix}_ffn_norm_weight", nn.Parameter(torch.ones(dim)))
        # Gamma (layer scale)
        self.register_parameter(f"{prefix}_gamma", nn.Parameter(torch.ones(dim)))
        self.register_parameter(f"{prefix}_ffn_gamma", nn.Parameter(torch.ones(dim)))
        # FFN: linear1 (dim -> 4*dim), linear2 (4*dim -> dim)
        ffn_dim = 4 * dim
        self.register_parameter(f"{prefix}_ffn_l1_weight", nn.Parameter(torch.zeros(ffn_dim, dim)))
        self.register_parameter(f"{prefix}_ffn_l1_bias", nn.Parameter(torch.zeros(ffn_dim)))
        self.register_parameter(f"{prefix}_ffn_l2_weight", nn.Parameter(torch.zeros(dim, ffn_dim)))
        self.register_parameter(f"{prefix}_ffn_l2_bias", nn.Parameter(torch.zeros(dim)))

    def _build_architecture(self):
        """Register all layers matching TokenizerEncoder structure."""
        # Stem
        self._register_conv("stem", 1, N_FILTERS, KERNEL_SIZE)

        # Stages + downsamples
        for i in range(len(DEPTHS)):
            ch = N_FILTERS * (2 ** i)
            # Stage blocks
            for j in range(DEPTHS[i]):
                self._register_block(i, j, ch)
            # Downsample (except after last stage)
            if i < len(RATIOS):
                ratio = RATIOS[i]
                out_ch = ch * 2
                self._register_conv(f"downsample{i}", ch, out_ch, ratio * 2, stride=ratio)

        # Final norm is Identity (disable_last_norm=True in VibeVoice)
        last_ch = N_FILTERS * (2 ** len(RATIOS))

        # Head
        self._register_conv("head", last_ch, OUTPUT_DIM, LAST_KERNEL_SIZE)

    def _apply_conv(self, x, name, stride, groups, ctx, in_ch):
        """Apply a single causal conv with cache read/update."""
        weight = getattr(self, f"{name}_weight")
        bias = getattr(self, f"{name}_bias")
        cache = getattr(self, f"{name}_cache")

        if ctx > 0:
            x_with_ctx = torch.cat([cache, x], dim=2)
        else:
            x_with_ctx = x

        out = F.conv1d(x_with_ctx, weight, bias, stride=stride, groups=groups)

        # Update cache: last ctx samples of input (before conv)
        if ctx > 0:
            total_len = x_with_ctx.shape[2]
            new_cache = x_with_ctx[:, :, total_len - ctx:]
            setattr(self, f"{name}_cache", new_cache)

        return out

    def _apply_rms_norm(self, x, weight_name):
        """RMS norm on channel dimension (input is BCT)."""
        weight = getattr(self, weight_name)
        # Transpose to BTC for norm, then back
        xt = x.transpose(1, 2)  # BTC
        variance = xt.pow(2).mean(-1, keepdim=True)
        xt = xt * torch.rsqrt(variance + RMS_EPS)
        xt = xt * weight
        return xt.transpose(1, 2)  # BCT

    def _apply_block(self, x, stage_idx, block_idx):
        """Apply a Block1D: norm → depthwise conv → gamma → residual, then FFN."""
        prefix = f"stage{stage_idx}_block{block_idx}"
        conv_name = f"{prefix}_conv"
        # Find conv info
        for name, stride, groups, ctx, in_ch in self._conv_layers:
            if name == conv_name:
                break

        # Mixer
        residual = x
        x = self._apply_rms_norm(x, f"{prefix}_norm_weight")
        x = self._apply_conv(x, conv_name, stride, groups, ctx, in_ch)
        gamma = getattr(self, f"{prefix}_gamma")
        x = x * gamma.unsqueeze(-1)
        x = residual + x

        # FFN
        residual = x
        x = self._apply_rms_norm(x, f"{prefix}_ffn_norm_weight")
        xt = x.transpose(1, 2)  # BTC
        l1_w = getattr(self, f"{prefix}_ffn_l1_weight")
        l1_b = getattr(self, f"{prefix}_ffn_l1_bias")
        l2_w = getattr(self, f"{prefix}_ffn_l2_weight")
        l2_b = getattr(self, f"{prefix}_ffn_l2_bias")
        xt = F.gelu(F.linear(xt, l1_w, l1_b))
        xt = F.linear(xt, l2_w, l2_b)
        x = xt.transpose(1, 2)  # BCT
        ffn_gamma = getattr(self, f"{prefix}_ffn_gamma")
        x = x * ffn_gamma.unsqueeze(-1)
        x = residual + x

        return x

    def forward(self, audio):
        """Process one frame of audio (3200 samples) using cached conv state.

        Args:
            audio: [1, 1, 3200] float32

        Returns:
            features: [1, OUTPUT_DIM, 1] float32
        """
        x = audio

        # Stem
        x = self._apply_conv(x, "stem", 1, 1, _context_size(KERNEL_SIZE), 1)

        # Stages + downsamples
        for i in range(len(DEPTHS)):
            for j in range(DEPTHS[i]):
                x = self._apply_block(x, i, j)
            if i < len(RATIOS):
                ratio = RATIOS[i]
                ctx = _context_size(ratio * 2, ratio)
                ch = N_FILTERS * (2 ** i)
                x = self._apply_conv(x, f"downsample{i}", ratio, 1, ctx, ch)

        # Final norm is Identity (skipped)

        # Head
        last_ch = N_FILTERS * (2 ** len(RATIOS))
        x = self._apply_conv(x, "head", 1, 1, _context_size(LAST_KERNEL_SIZE), last_ch)

        return x


# ─── Weight loading ──────────────────────────────────────────────────────────

def load_weights(model: StatefulStreamingSemanticEncoder, encoder: nn.Module):
    """Copy weights from the original TokenizerEncoder into our stateful model."""
    sd = {}

    # Stem: downsample_layers[0][0] is SConv1d
    stem_conv = encoder.downsample_layers[0][0]
    sd["stem_weight"] = stem_conv.conv.conv.weight.data.float()
    sd["stem_bias"] = stem_conv.conv.conv.bias.data.float()

    # Stages + downsamples
    for i in range(len(DEPTHS)):
        ch = N_FILTERS * (2 ** i)

        # Stage blocks
        for j in range(DEPTHS[i]):
            block = encoder.stages[i][j]
            prefix = f"stage{i}_block{j}"

            # Depthwise conv
            sd[f"{prefix}_conv_weight"] = block.mixer.conv.conv.conv.weight.data.float()
            sd[f"{prefix}_conv_bias"] = block.mixer.conv.conv.conv.bias.data.float()

            # Norms
            sd[f"{prefix}_norm_weight"] = block.norm.weight.data.float()
            sd[f"{prefix}_ffn_norm_weight"] = block.ffn_norm.weight.data.float()

            # Gamma
            if block.gamma is not None:
                sd[f"{prefix}_gamma"] = block.gamma.data.float()
            if block.ffn_gamma is not None:
                sd[f"{prefix}_ffn_gamma"] = block.ffn_gamma.data.float()

            # FFN
            sd[f"{prefix}_ffn_l1_weight"] = block.ffn.linear1.weight.data.float()
            sd[f"{prefix}_ffn_l1_bias"] = block.ffn.linear1.bias.data.float()
            sd[f"{prefix}_ffn_l2_weight"] = block.ffn.linear2.weight.data.float()
            sd[f"{prefix}_ffn_l2_bias"] = block.ffn.linear2.bias.data.float()

        # Downsample
        if i < len(RATIOS):
            ds_conv = encoder.downsample_layers[i + 1][0]  # +1 because [0] is stem
            sd[f"downsample{i}_weight"] = ds_conv.conv.conv.weight.data.float()
            sd[f"downsample{i}_bias"] = ds_conv.conv.conv.bias.data.float()

    # Final norm is Identity — no weights

    # Head
    sd["head_weight"] = encoder.head.conv.conv.weight.data.float()
    sd["head_bias"] = encoder.head.conv.conv.bias.data.float()

    # Load
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only cache buffers should be missing (they're zero-initialized)
    real_missing = [k for k in missing if not k.endswith("_cache")]
    if real_missing:
        print(f"  WARNING: missing weights: {real_missing}")
    if unexpected:
        print(f"  WARNING: unexpected weights: {unexpected}")
    print(f"  Loaded {len(sd)} parameters ({len(missing)} cache buffers kept at zero)")


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_streaming(model: StatefulStreamingSemanticEncoder, encoder: nn.Module, num_frames=5):
    """Compare streaming output against non-streaming (full-sequence) encoder."""
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

    torch.manual_seed(42)
    total_samples = FRAME_SAMPLES * num_frames
    audio_full = torch.randn(1, 1, total_samples)

    # Non-streaming: full encoder
    with torch.no_grad():
        ref_features = encoder(audio_full)  # [1, 128, num_frames]

    # Streaming: our stateful model, frame by frame
    model.eval()
    # Reset caches to zero
    for name in list(model._buffers.keys()):
        if name.endswith("_cache"):
            model._buffers[name] = torch.zeros_like(model._buffers[name])

    streaming_frames = []
    with torch.no_grad():
        for f in range(num_frames):
            chunk = audio_full[:, :, f * FRAME_SAMPLES:(f + 1) * FRAME_SAMPLES]
            out = model(chunk)  # [1, 128, 1]
            streaming_frames.append(out)

    streaming_features = torch.cat(streaming_frames, dim=2)  # [1, 128, num_frames]

    # Also validate against PyTorch streaming cache
    cache = VibeVoiceTokenizerStreamingCache()
    pt_streaming_frames = []
    with torch.no_grad():
        for f in range(num_frames):
            chunk = audio_full[:, :, f * FRAME_SAMPLES:(f + 1) * FRAME_SAMPLES]
            out = encoder(chunk, cache=cache, sample_indices=torch.tensor([0]), use_cache=True)
            pt_streaming_frames.append(out)
    pt_streaming = torch.cat(pt_streaming_frames, dim=2)

    print(f"\n  Validation ({num_frames} frames):")
    diff_ns = (ref_features - streaming_features).abs()
    diff_pt = (pt_streaming - streaming_features).abs()
    print(f"    vs non-streaming:   max={diff_ns.max():.6e}  mean={diff_ns.mean():.6e}")
    print(f"    vs PyTorch stream:  max={diff_pt.max():.6e}  mean={diff_pt.mean():.6e}")
    ok = diff_ns.max() < 1e-4 and diff_pt.max() < 1e-4
    print(f"    {'OK' if ok else 'MISMATCH'}")
    return ok


# ─── CoreML conversion ───────────────────────────────────────────────────────

def convert_to_coreml(model: StatefulStreamingSemanticEncoder, output_path: Path):
    """Trace and convert to CoreML with state tensors for conv caches."""
    import coremltools as ct

    model.eval()
    trace_input = torch.randn(1, 1, FRAME_SAMPLES)

    print("  Tracing...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, (trace_input,))
    traced.eval()
    print(f"  Traced in {time.time() - t0:.1f}s")

    # Build CoreML inputs/outputs
    inputs = [ct.TensorType("audio", shape=(1, 1, FRAME_SAMPLES), dtype=np.float32)]
    outputs = [ct.TensorType("features", dtype=np.float32)]

    # Build state tensors for all conv caches
    states = []
    for name, stride, groups, ctx, in_ch in model._conv_layers:
        if ctx > 0:
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, in_ch, ctx),
                    dtype=np.float16,
                ),
                name=f"{name}_cache",
            ))

    print(f"  Converting to CoreML ({len(states)} state buffers)...")
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
    print(f"  Converted in {time.time() - t0:.1f}s")

    mlmodel.save(str(output_path))
    print(f"  Saved: {output_path}")

    # Quick validation
    print("  Validating CoreML...")
    state = mlmodel.make_state()
    test_audio = np.random.randn(1, 1, FRAME_SAMPLES).astype(np.float32)
    out = mlmodel.predict({"audio": test_audio}, state=state)
    print(f"  Output: {out['features'].shape} — OK")

    return mlmodel


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export streaming semantic encoder to CoreML")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUILD_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Streaming Semantic Encoder Export ===\n")

    # Load original model
    print("Loading VibeVoice model...")
    t0 = time.time()
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
    vv = VibeVoiceForConditionalGeneration.from_pretrained(
        "microsoft/VibeVoice-1.5B", torch_dtype=torch.float32
    )
    vv.eval()
    encoder = vv.model.semantic_tokenizer.encoder
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Build stateful model and load weights
    print("\nBuilding stateful streaming encoder...")
    streaming_model = StatefulStreamingSemanticEncoder()
    load_weights(streaming_model, encoder)

    # Validate
    print("\nValidating...")
    ok = validate_streaming(streaming_model, encoder)
    if not ok:
        print("\n  VALIDATION FAILED — aborting export")
        return

    # Convert to CoreML
    print("\nConverting to CoreML...")
    output_path = output_dir / "semantic_encoder_streaming.mlpackage"
    convert_to_coreml(streaming_model, output_path)

    new_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"\n  Size: {new_size / 1e6:.1f}MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
