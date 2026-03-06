#!/usr/bin/env python3
"""Convert VibeVoice-ASR-HF Qwen2 backbone to stateful CoreML.

The ASR model uses a Qwen2 decoder (28 layers, hidden=3584) that generates
text tokens autoregressively after receiving audio embeddings. This exports
just the decoder transformer with stateful KV cache.

Pipeline at runtime:
  1. Audio -> acoustic_encoder + semantic_encoder -> features
  2. Features -> connectors -> combined embeddings
  3. Embeddings inserted into text prompt -> embed_tokens lookup
  4. Combined -> THIS MODEL (stateful decoder) -> hidden states
  5. Hidden states -> lm_head -> next token logits

Usage:
    uv run python convert_stateful_lm.py --output-dir ./build/vibevoice-asr-hf
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

# Qwen2 architecture for ASR-HF
NUM_LAYERS = 28
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128  # 3584 / 28
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
VOCAB_SIZE = 152064
ROPE_THETA = 1000000.0
RMS_NORM_EPS = 1e-6
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 7


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
    """Qwen2 decoder with stateful KV cache for ASR."""

    def __init__(self, layers: nn.ModuleList, final_norm: nn.Module, max_seq_len: int):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

        for i in range(NUM_LAYERS):
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
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len

        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)

        for i in range(NUM_LAYERS):
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

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice-ASR Qwen2 to stateful CoreML")
    parser.add_argument("--model-id", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--output-dir", default="build/vibevoice-asr-hf")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print(f"Architecture: {NUM_LAYERS}L, h={HIDDEN_SIZE}, {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV")
    print(f"Max seq len: {MAX_SEQ_LEN}")

    # ---- Load weights ----
    print(f"\nLoading weights...")
    t0 = time.time()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    text_config = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        hidden_act="silu",
    )

    text_model = Qwen2Model(text_config)
    text_model.eval()

    model_dir = snapshot_download(args.model_id, allow_patterns=["*.safetensors", "*.json"])
    safetensor_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))

    all_weights = {}
    for f in safetensor_files:
        all_weights.update(load_file(f))

    # ASR-HF key prefix: model.language_model.X
    decoder_weights = {}
    for k, v in all_weights.items():
        if k.startswith("model.language_model."):
            decoder_weights[k[len("model.language_model."):]] = v.float()

    missing, _ = text_model.load_state_dict(decoder_weights, strict=False)
    print(f"Loaded in {time.time() - t0:.1f}s ({len(decoder_weights)} tensors)")
    del all_weights

    # ---- Create stateful model ----
    layers = text_model.layers
    final_norm = text_model.norm

    print(f"\nCreating stateful decoder...")
    stateful_model = StatefulQwen2Decoder(layers, final_norm, MAX_SEQ_LEN)
    stateful_model.eval()

    # ---- Trace ----
    hidden = torch.randn(1, 1, HIDDEN_SIZE)
    cos_in = torch.randn(1, 1, HEAD_DIM)
    sin_in = torch.randn(1, 1, HEAD_DIM)
    mask = torch.zeros(1, 1, 1, 5)

    print("Tracing...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(stateful_model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"Done in {time.time() - t0:.1f}s")

    # ---- CoreML convert ----
    print("\nConverting to CoreML...")
    import coremltools as ct

    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [ct.TensorType("output_hidden", dtype=np.float32)]

    states = []
    for i in range(NUM_LAYERS):
        for prefix in ("k", "v"):
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM),
                    dtype=np.float16,
                ),
                name=f"{prefix}_cache_{i}",
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
    print(f"Conversion in {time.time() - t0:.1f}s")

    output_path = output_dir / "lm_decoder_stateful.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Saved: {output_path}")

    # ---- Export embeddings ----
    embed_weights = text_model.embed_tokens.weight.detach().float().numpy()
    embed_path = output_dir / "embed_tokens.npy"
    np.save(str(embed_path), embed_weights)
    print(f"Embeddings: {embed_path} {embed_weights.shape}")

    # ---- Validate ----
    print("\nValidating...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
            "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        output = mlmodel.predict(test_input, state=state)
        print(f"  Output: {output['output_hidden'].shape} — OK")
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
