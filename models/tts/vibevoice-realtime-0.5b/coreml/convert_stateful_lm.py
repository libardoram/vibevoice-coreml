#!/usr/bin/env python3
"""Convert VibeVoice-Realtime-0.5B LLM backbones to stateful CoreML.

The streaming model splits Qwen2.5-0.5B into two separate LMs:
  - language_model: 4 layers (base text encoding, no final norm)
  - tts_language_model: 20 layers (TTS generation, with final norm)

Each has its own KV cache. We export them as two stateful CoreML models.

Also exports:
  - embed_tokens.npy — shared embedding table
  - tts_input_types.npy — 2-entry type embedding (text=1, speech=0)
  - speech_scaling_factor, speech_bias_factor — normalization constants

Usage:
    uv run python convert_stateful_lm.py --output-dir ./build/vibevoice-realtime-0.5b
"""

import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen2.5-0.5B architecture constants
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64  # 896 / 14
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
VOCAB_SIZE = 151936
ROPE_THETA = 1000000.0
RMS_NORM_EPS = 1e-6
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 7

# Split: 24 total = 4 base + 20 TTS (tts_backbone_num_hidden_layers=20)
BASE_LM_LAYERS = 4
TTS_LM_LAYERS = 20


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


class StatefulQwen2Decoder(nn.Module):
    """Qwen2.5 decoder subset with stateful KV cache.

    Parameterized by number of layers — used for both the 4-layer base LM
    and the 20-layer TTS LM.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        final_norm: nn.Module,
        num_layers: int,
        max_seq_len: int,
        apply_final_norm: bool = True,
    ):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.apply_final_norm = apply_final_norm
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

        for i in range(num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [1, Q, 896]
            position_cos:  [1, Q, 64]
            position_sin:  [1, Q, 64]
            attention_mask: [1, 1, Q, end_step]

        Returns:
            output_hidden: [1, Q, 896]
        """
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

            q = q.view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            k_full = k_cache[:, :, :end_step, :].float()
            v_full = v_cache[:, :, :end_step, :].float()

            k_full = repeat_kv(k_full, GQA_REPEAT)
            v_full = repeat_kv(v_full, GQA_REPEAT)

            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(1, q_len, NUM_Q_HEADS * HEAD_DIM)
            hidden_states = attn.o_proj(attn_output)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            hidden_states = mlp.down_proj(F.silu(gate) * up)
            hidden_states = residual + hidden_states

        if self.apply_final_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


def _trace_and_convert(
    model: StatefulQwen2Decoder,
    name: str,
    num_layers: int,
    max_seq_len: int,
    output_dir: Path,
):
    """Trace, validate, convert to CoreML, and save."""
    import coremltools as ct

    # Trace
    trace_q = 1
    trace_end = 5
    hidden = torch.randn(1, trace_q, HIDDEN_SIZE)
    cos_in = torch.randn(1, trace_q, HEAD_DIM)
    sin_in = torch.randn(1, trace_q, HEAD_DIM)
    mask = torch.zeros(1, 1, trace_q, trace_end)

    print(f"  Tracing {name} ({num_layers} layers)...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"  Traced in {time.time() - t0:.1f}s")

    # Validate
    ref_model = StatefulQwen2Decoder(
        model.layers, model.final_norm, num_layers, max_seq_len,
        apply_final_norm=model.apply_final_norm,
    )
    ref_model.eval()
    th = torch.randn(1, 1, HIDDEN_SIZE)
    tc = torch.randn(1, 1, HEAD_DIM)
    ts = torch.randn(1, 1, HEAD_DIM)
    tm = torch.zeros(1, 1, 1, 3)
    with torch.no_grad():
        diff = (ref_model(th, tc, ts, tm) - traced(th, tc, ts, tm)).abs().max().item()
    print(f"  Validation diff: {diff:.6e} {'OK' if diff < 1e-3 else 'WARNING'}")

    # CoreML convert
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [ct.TensorType("output_hidden", dtype=np.float32)]

    states = []
    for i in range(num_layers):
        for prefix in ("k", "v"):
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM),
                    dtype=np.float16,
                ),
                name=f"{prefix}_cache_{i}",
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

    path = output_dir / f"{name}.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path}")

    # Quick validation
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
            "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        output = mlmodel.predict(test_input, state=state)
        print(f"  CoreML output: {output['output_hidden'].shape} — OK")
    except Exception as e:
        print(f"  CoreML validation failed: {e}")

    return path


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice-Realtime-0.5B LMs to stateful CoreML")
    parser.add_argument("--model-id", default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--output-dir", default="build/vibevoice-realtime-0.5b")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print(f"Architecture: {BASE_LM_LAYERS} base + {TTS_LM_LAYERS} TTS layers")
    print(f"Max seq len: {MAX_SEQ_LEN}")

    # ---- Load weights ----
    print(f"\nLoading weights...")
    t0 = time.time()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    model_dir = snapshot_download(args.model_id, allow_patterns=["*.safetensors", "*.json"])
    safetensor_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))

    all_weights = {}
    for f in safetensor_files:
        all_weights.update(load_file(f))

    # ---- Build base LM (4 layers, no final norm) ----
    base_config = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=BASE_LM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        hidden_act="silu",
    )
    base_model = Qwen2Model(base_config).eval()

    base_weights = {}
    for k, v in all_weights.items():
        if k.startswith("model.language_model."):
            base_weights[k[len("model.language_model."):]] = v.float()
    missing, _ = base_model.load_state_dict(base_weights, strict=False)
    print(f"Base LM: {len(base_weights)} tensors loaded, {len(missing)} missing")

    # ---- Build TTS LM (20 layers, with final norm) ----
    tts_config = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=TTS_LM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        hidden_act="silu",
    )
    tts_model = Qwen2Model(tts_config).eval()

    tts_weights = {}
    for k, v in all_weights.items():
        if k.startswith("model.tts_language_model."):
            tts_weights[k[len("model.tts_language_model."):]] = v.float()
    missing, _ = tts_model.load_state_dict(tts_weights, strict=False)
    print(f"TTS LM: {len(tts_weights)} tensors loaded, {len(missing)} missing")

    print(f"Loaded in {time.time() - t0:.1f}s")

    # ---- Export base LM (no final norm — uses nn.Identity in original) ----
    print("\n--- Base LM ---")
    base_stateful = StatefulQwen2Decoder(
        base_model.layers, base_model.norm, BASE_LM_LAYERS, MAX_SEQ_LEN,
        apply_final_norm=False,  # Original uses nn.Identity
    )
    base_stateful.eval()
    _trace_and_convert(base_stateful, "base_lm_stateful", BASE_LM_LAYERS, MAX_SEQ_LEN, output_dir)

    # ---- Export TTS LM (with final norm) ----
    print("\n--- TTS LM ---")
    tts_stateful = StatefulQwen2Decoder(
        tts_model.layers, tts_model.norm, TTS_LM_LAYERS, MAX_SEQ_LEN,
        apply_final_norm=True,
    )
    tts_stateful.eval()
    _trace_and_convert(tts_stateful, "tts_lm_stateful", TTS_LM_LAYERS, MAX_SEQ_LEN, output_dir)

    # ---- Export embeddings and constants ----
    print("\n--- Constants ---")

    # Shared embedding table
    embed_weights = base_model.embed_tokens.weight.detach().float().numpy()
    np.save(str(output_dir / "embed_tokens.npy"), embed_weights)
    print(f"  embed_tokens: {embed_weights.shape}")

    # TTS type embeddings (text=1, speech=0)
    tts_type_key = "model.tts_input_types.weight"
    if tts_type_key in all_weights:
        tts_types = all_weights[tts_type_key].float().numpy()
        np.save(str(output_dir / "tts_input_types.npy"), tts_types)
        print(f"  tts_input_types: {tts_types.shape}")
    else:
        print(f"  WARNING: {tts_type_key} not found in weights")

    # Speech scaling/bias factors
    for name in ("model.speech_scaling_factor", "model.speech_bias_factor"):
        if name in all_weights:
            val = all_weights[name].float().item()
            short_name = name.split(".")[-1]
            np.save(str(output_dir / f"{short_name}.npy"), np.array([val], dtype=np.float32))
            print(f"  {short_name}: {val}")

    del all_weights
    print("\nDone!")


if __name__ == "__main__":
    main()
