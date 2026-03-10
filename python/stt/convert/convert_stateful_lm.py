#!/usr/bin/env python3
"""Convert VibeVoice-ASR Qwen2-7B decoder to stateful CoreML.

Same Qwen2-7B backbone as TTS 7B: 28 layers, h=3584, 28Q/4KV heads.
The LM generates text tokens autoregressively from audio embeddings.

Exports:
  - lm_decoder_stateful.mlpackage — Qwen2 decoder with internal KV cache

Usage:
    uv run python convert/convert_stateful_lm.py --output-dir build/vibevoice-asr
"""

import argparse
import glob
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_common import rotate_half, repeat_kv, build_kv_state_specs

# Qwen2-7B config (same as TTS 7B)
CFG = {
    "num_layers": 28,
    "num_q_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,  # 3584 / 28
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "default_max_seq": 32768,
}


class StatefulQwen2Decoder(nn.Module):
    """Qwen2 decoder with stateful KV cache — same as TTS version."""

    def __init__(self, layers, final_norm, cfg, max_seq_len):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.num_layers = cfg["num_layers"]
        self.num_q_heads = cfg["num_q_heads"]
        self.num_kv_heads = cfg["num_kv_heads"]
        self.head_dim = cfg["head_dim"]
        self.hidden_size = cfg["hidden_size"]
        self.gqa_repeat = cfg["num_q_heads"] // cfg["num_kv_heads"]
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(cfg["head_dim"])

        for i in range(self.num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, dtype=torch.float16),
            )

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len

        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

            q = q.view(1, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            k_full = k_cache[:, :, :end_step, :].float()
            v_full = v_cache[:, :, :end_step, :].float()

            k_full = repeat_kv(k_full, self.gqa_repeat)
            v_full = repeat_kv(v_full, self.gqa_repeat)

            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(1, q_len, self.num_q_heads * self.head_dim)
            hidden_states = attn.o_proj(attn_output)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            hidden_states = mlp.down_proj(F.silu(gate) * up)
            hidden_states = residual + hidden_states

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice-ASR LM to stateful CoreML")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cfg = CFG
    MAX_SEQ_LEN = cfg["default_max_seq"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: microsoft/VibeVoice-ASR-HF")
    print(f"Config: {cfg['num_layers']}L, h={cfg['hidden_size']}, "
          f"{cfg['num_q_heads']}Q/{cfg['num_kv_heads']}KV, head_dim={cfg['head_dim']}")
    print(f"Max seq len: {MAX_SEQ_LEN}")

    # Load weights
    print(f"\nLoading model weights...")
    t0 = time.time()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

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

    model_dir = snapshot_download("microsoft/VibeVoice-ASR-HF",
                                  allow_patterns=["*.safetensors", "*.json"])
    safetensor_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))

    all_weights = {}
    for f in safetensor_files:
        all_weights.update(load_file(f))

    # ASR HF model uses "language_model.model." prefix for decoder layers
    decoder_weights = {}
    for k, v in all_weights.items():
        if k.startswith("language_model.model."):
            decoder_weights[k[len("language_model.model."):]] = v.float()

    missing, _ = text_model.load_state_dict(decoder_weights, strict=False)
    print(f"Loaded in {time.time() - t0:.1f}s ({len(decoder_weights)} tensors, {len(missing)} missing)")

    del all_weights, decoder_weights

    # Save embeddings
    embed_weights = text_model.embed_tokens.weight.detach().float().numpy()

    # Create stateful model
    print(f"\nCreating stateful decoder...")
    stateful_model = StatefulQwen2Decoder(text_model.layers, text_model.norm, cfg, MAX_SEQ_LEN)
    stateful_model.eval()

    # Trace
    hidden = torch.randn(1, 1, cfg["hidden_size"])
    cos_in = torch.randn(1, 1, cfg["head_dim"])
    sin_in = torch.randn(1, 1, cfg["head_dim"])
    mask = torch.zeros(1, 1, 1, 5)

    print("Tracing...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(stateful_model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"Done in {time.time() - t0:.1f}s")

    # Validate
    if not args.skip_validation:
        print("\nValidating trace...")
        th = torch.randn(1, 1, cfg["hidden_size"])
        tc = torch.randn(1, 1, cfg["head_dim"])
        ts = torch.randn(1, 1, cfg["head_dim"])
        tm = torch.zeros(1, 1, 1, 3)
        with torch.no_grad():
            out_orig = stateful_model(th, tc, ts, tm)
            out_traced = traced(th, tc, ts, tm)
            diff = (out_orig - out_traced).abs().max().item()
        print(f"  Max diff: {diff:.6e} {'OK' if diff < 1e-3 else 'WARNING'}")

    # Free PyTorch memory
    del stateful_model, text_model
    import gc
    gc.collect()
    print("Freed PyTorch model memory")

    # CoreML convert
    print("\nConverting to CoreML...")
    import coremltools as ct

    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, cfg["hidden_size"]), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, cfg["head_dim"]), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, cfg["head_dim"]), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [ct.TensorType("output_hidden", dtype=np.float32)]
    states = build_kv_state_specs(cfg["num_layers"], cfg["num_kv_heads"],
                                  MAX_SEQ_LEN, cfg["head_dim"])

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
    print(f"CoreML conversion in {time.time() - t0:.1f}s")

    output_path = output_dir / "lm_decoder_stateful.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Saved: {output_path}")

    # Save embeddings as float16 binary
    embed_f16 = embed_weights.astype(np.float16)
    embed_path = output_dir / "embed_tokens.bin"
    with open(embed_path, "wb") as f:
        vocab_size, hidden_size = embed_weights.shape
        f.write(np.array([vocab_size, hidden_size], dtype=np.uint32).tobytes())
        f.write(embed_f16.tobytes())
    f16_mb = embed_path.stat().st_size / 1e6
    print(f"Embeddings: {embed_path} {embed_weights.shape} ({f16_mb:.1f}MB)")

    # CoreML validation
    print("\nValidating CoreML...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, cfg["hidden_size"]).astype(np.float32),
            "position_cos": np.random.randn(1, 1, cfg["head_dim"]).astype(np.float32),
            "position_sin": np.random.randn(1, 1, cfg["head_dim"]).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        output = mlmodel.predict(test_input, state=state)
        print(f"  Output: {output['output_hidden'].shape} — OK")
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
