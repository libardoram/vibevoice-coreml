"""Shared MLX component implementations for benchmarking.

Pure MLX reimplementations of VibeVoice components, loading weights
directly from safetensors. Used by per-model bench_mlx.py scripts.
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn


# ─── Utilities ────────────────────────────────────────────────────────────────

def rms_norm(x, eps, weight=None):
    return mx.fast.rms_norm(x, weight if weight is not None else mx.ones(x.shape[-1], dtype=x.dtype), eps)


def to_mx(t, dtype=mx.float16):
    return mx.array(t.float().numpy()).astype(dtype)


class QuantizedWeight:
    """INT8 quantized weight matrix using mx.quantize/mx.quantized_matmul."""
    def __init__(self, w_fp, group_size=32, bits=8):
        self.w_q, self.scales, self.biases = mx.quantize(w_fp, group_size=group_size, bits=bits)
        self.group_size = group_size
        self.bits = bits
        self.shape = w_fp.shape


def qmm(x, w):
    """x @ w.T — works for both plain mx.array and QuantizedWeight."""
    if isinstance(w, QuantizedWeight):
        return mx.quantized_matmul(
            x, w.w_q, w.scales, w.biases,
            transpose=True, group_size=w.group_size, bits=w.bits,
        )
    return x @ w.T


def to_mx_q(t, dtype=mx.float16, quantize=False, group_size=32, bits=8):
    """Load weight, optionally quantize."""
    w = mx.array(t.float().numpy()).astype(dtype)
    if quantize and w.ndim >= 2 and w.shape[-1] % group_size == 0:
        return QuantizedWeight(w, group_size=group_size, bits=bits)
    return w


def benchmark(fn, warmup=10, iterations=100, label=""):
    for _ in range(warmup):
        fn()
        mx.eval(mx.zeros(1))
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        mx.eval(mx.zeros(1))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    med = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mn = times[0]
    if label:
        print(f"  {label}: {med:.2f}ms median, {p95:.2f}ms p95, {mn:.2f}ms min")
    return {"median_ms": med, "p95_ms": p95, "min_ms": mn}


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def make_rope(pos, head_dim, rope_theta, dtype=mx.float16):
    inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = pos * inv_freq
    freqs = mx.concatenate([freqs, freqs])
    cos = mx.cos(freqs).reshape(1, 1, 1, head_dim).astype(dtype)
    sin = mx.sin(freqs).reshape(1, 1, 1, head_dim).astype(dtype)
    return cos, sin


# ─── Diffusion Head ───────────────────────────────────────────────────────────

class TimestepEmbedder:
    def __init__(self, w0, w2, freq_dim=256):
        self.w0 = w0
        self.w2 = w2
        self.freq_dim = freq_dim

    def __call__(self, t):
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(self.w0.dtype)
        return nn.silu(emb @ self.w0.T) @ self.w2.T


class HeadLayer:
    def __init__(self, adaln_w, norm_w, gate_w, up_w, down_w):
        self.adaln_w = adaln_w
        self.norm_w = norm_w
        self.gate_w = gate_w
        self.up_w = up_w
        self.down_w = down_w

    def __call__(self, x, c):
        mods = nn.silu(c) @ self.adaln_w.T
        H = x.shape[-1]
        shift, scale, gate = mods[:, :H], mods[:, H:2*H], mods[:, 2*H:]
        h = rms_norm(x, 1e-5, self.norm_w)
        h = h * (1 + scale) + shift
        h = self.down_w @ (nn.silu(h @ self.gate_w.T) * (h @ self.up_w.T)).T
        h = h.T
        return x + gate * h


class FinalLayer:
    def __init__(self, adaln_w, linear_w):
        self.adaln_w = adaln_w
        self.linear_w = linear_w

    def __call__(self, x, c):
        mods = nn.silu(c) @ self.adaln_w.T
        H = x.shape[-1]
        shift, scale = mods[:, :H], mods[:, H:]
        h = rms_norm(x, 1e-5)
        h = h * (1 + scale) + shift
        return h @ self.linear_w.T


class DiffusionHead:
    def __init__(self, weights, dtype=mx.float16):
        p = "model.prediction_head."
        self.noisy_proj = to_mx(weights[p + "noisy_images_proj.weight"], dtype)
        self.cond_proj = to_mx(weights[p + "cond_proj.weight"], dtype)
        self.t_emb = TimestepEmbedder(
            to_mx(weights[p + "t_embedder.mlp.0.weight"], dtype),
            to_mx(weights[p + "t_embedder.mlp.2.weight"], dtype),
        )
        self.layers = []
        for i in range(4):
            lp = f"{p}layers.{i}."
            self.layers.append(HeadLayer(
                to_mx(weights[lp + "adaLN_modulation.1.weight"], dtype),
                to_mx(weights[lp + "norm.weight"], dtype),
                to_mx(weights[lp + "ffn.gate_proj.weight"], dtype),
                to_mx(weights[lp + "ffn.up_proj.weight"], dtype),
                to_mx(weights[lp + "ffn.down_proj.weight"], dtype),
            ))
        fp = f"{p}final_layer."
        self.final = FinalLayer(
            to_mx(weights[fp + "adaLN_modulation.1.weight"], dtype),
            to_mx(weights[fp + "linear.weight"], dtype),
        )

    def __call__(self, noisy, timestep, condition):
        return self._forward(noisy, timestep, condition)

    def _forward(self, noisy, timestep, condition):
        x = noisy @ self.noisy_proj.T
        t = self.t_emb(timestep)
        c = condition @ self.cond_proj.T + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final(x, c)

    def compile(self):
        """Compile the forward pass for faster execution."""
        self._forward = mx.compile(self._forward)


# ─── VAE Decoder ──────────────────────────────────────────────────────────────

class DecoderBlock:
    """RMSNorm -> depthwise conv1d -> layerscale + RMSNorm -> FFN -> layerscale"""
    def __init__(self, norm_w, conv_w, conv_b, gamma,
                 ffn_norm_w, ffn_l1_w, ffn_l1_b, ffn_l2_w, ffn_l2_b, ffn_gamma,
                 dtype=mx.float16):
        self.norm_w = norm_w
        self.conv_w = conv_w
        self.conv_b = conv_b
        self.gamma = gamma
        self.ffn_norm_w = ffn_norm_w
        self.ffn_l1_w = ffn_l1_w
        self.ffn_l1_b = ffn_l1_b
        self.ffn_l2_w = ffn_l2_w
        self.ffn_l2_b = ffn_l2_b
        self.ffn_gamma = ffn_gamma

    def __call__(self, x):
        B, C, T = x.shape
        residual = x
        xt = x.transpose(0, 2, 1)
        xt = rms_norm(xt, 1e-5, self.norm_w)
        x = xt.transpose(0, 2, 1)
        K = self.conv_w.shape[-1]
        pad = K - 1
        x = mx.pad(x, [(0, 0), (0, 0), (pad, 0)])
        x = mx.conv1d(x.transpose(0, 2, 1), self.conv_w.transpose(0, 2, 1),
                       groups=C).transpose(0, 2, 1) + self.conv_b[:, None]
        x = residual + self.gamma[:, None] * x
        residual = x
        xt = x.transpose(0, 2, 1)
        xt = rms_norm(xt, 1e-5, self.ffn_norm_w)
        xt = nn.gelu(xt @ self.ffn_l1_w.T + self.ffn_l1_b)
        xt = xt @ self.ffn_l2_w.T + self.ffn_l2_b
        x = residual + self.ffn_gamma[:, None] * xt.transpose(0, 2, 1)
        return x


class VAEDecoder:
    def __init__(self, weights, dtype=mx.float16):
        p = "model.acoustic_tokenizer.decoder."
        depths = [8, 3, 3, 3, 3, 3, 3]
        ratios = [8, 5, 5, 4, 2, 2]

        self.init_conv_w = to_mx(weights[p + "upsample_layers.0.0.conv.conv.weight"], dtype)
        self.init_conv_b = to_mx(weights[p + "upsample_layers.0.0.conv.conv.bias"], dtype)

        self.stages = []
        for s in range(7):
            blocks = []
            for b in range(depths[s]):
                bp = f"{p}stages.{s}.{b}."
                blocks.append(DecoderBlock(
                    to_mx(weights[bp + "norm.weight"], dtype),
                    to_mx(weights[bp + "mixer.conv.conv.conv.weight"], dtype),
                    to_mx(weights[bp + "mixer.conv.conv.conv.bias"], dtype),
                    to_mx(weights[bp + "gamma"], dtype),
                    to_mx(weights[bp + "ffn_norm.weight"], dtype),
                    to_mx(weights[bp + "ffn.linear1.weight"], dtype),
                    to_mx(weights[bp + "ffn.linear1.bias"], dtype),
                    to_mx(weights[bp + "ffn.linear2.weight"], dtype),
                    to_mx(weights[bp + "ffn.linear2.bias"], dtype),
                    to_mx(weights[bp + "ffn_gamma"], dtype),
                    dtype=dtype,
                ))
            self.stages.append(blocks)

        self.upsample_convs = []
        for i in range(1, 7):
            up = f"{p}upsample_layers.{i}.0.convtr.convtr."
            self.upsample_convs.append((
                to_mx(weights[up + "weight"], dtype),
                to_mx(weights[up + "bias"], dtype),
                ratios[i - 1],
            ))

        self.head_w = to_mx(weights[p + "head.conv.conv.weight"], dtype)
        self.head_b = to_mx(weights[p + "head.conv.conv.bias"], dtype)

    def __call__(self, latent):
        K = self.init_conv_w.shape[-1]
        x = mx.pad(latent, [(0, 0), (0, 0), (K - 1, 0)])
        x = mx.conv1d(
            x.transpose(0, 2, 1),
            self.init_conv_w.transpose(0, 2, 1),
        ).transpose(0, 2, 1) + self.init_conv_b[:, None]

        for s_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)
            if s_idx < len(self.upsample_convs):
                w, b, stride = self.upsample_convs[s_idx]
                x = mx.conv_transpose1d(
                    x.transpose(0, 2, 1),
                    w.transpose(1, 2, 0),
                    stride=stride,
                ).transpose(0, 2, 1) + b[:, None]
                K_up = w.shape[-1]
                trim = (K_up - stride) // 2
                if trim > 0:
                    x = x[:, :, trim:-trim]

        K = self.head_w.shape[-1]
        x = mx.pad(x, [(0, 0), (0, 0), (K - 1, 0)])
        x = mx.conv1d(
            x.transpose(0, 2, 1),
            self.head_w.transpose(0, 2, 1),
        ).transpose(0, 2, 1) + self.head_b[:, None]
        return x


# ─── Connectors ──────────────────────────────────────────────────────────────

class AcousticConnector:
    def __init__(self, weights, dtype=mx.float16):
        p = "model.acoustic_connector."
        self.fc1_w = to_mx(weights[p + "fc1.weight"], dtype)
        self.fc1_b = to_mx(weights[p + "fc1.bias"], dtype)
        self.norm_w = to_mx(weights[p + "norm.weight"], dtype)
        self.fc2_w = to_mx(weights[p + "fc2.weight"], dtype)
        self.fc2_b = to_mx(weights[p + "fc2.bias"], dtype)

    def __call__(self, x):
        x = x @ self.fc1_w.T + self.fc1_b
        x = rms_norm(x, 1e-5, self.norm_w)
        return x @ self.fc2_w.T + self.fc2_b


class SemanticConnector:
    def __init__(self, weights, dtype=mx.float16):
        p = "model.semantic_connector."
        self.fc1_w = to_mx(weights[p + "fc1.weight"], dtype)
        self.fc1_b = to_mx(weights[p + "fc1.bias"], dtype)
        self.norm_w = to_mx(weights[p + "norm.weight"], dtype)
        self.fc2_w = to_mx(weights[p + "fc2.weight"], dtype)
        self.fc2_b = to_mx(weights[p + "fc2.bias"], dtype)

    def __call__(self, x):
        x = x @ self.fc1_w.T + self.fc1_b
        x = rms_norm(x, 1e-5, self.norm_w)
        return x @ self.fc2_w.T + self.fc2_b


# ─── LM Head ─────────────────────────────────────────────────────────────────

class LMHead:
    def __init__(self, weights, key="model.language_model.embed_tokens.weight", dtype=mx.float16, quantize=False):
        self.weight = to_mx_q(weights[key], dtype, quantize=quantize)

    def __call__(self, x):
        # Compute in float32 to prevent fp16 overflow with untied lm_head (7B)
        if isinstance(self.weight, QuantizedWeight):
            return mx.quantized_matmul(
                x.astype(mx.float32),
                self.weight.w_q, self.weight.scales, self.weight.biases,
                transpose=True, group_size=self.weight.group_size, bits=self.weight.bits,
            ).astype(x.dtype)
        return (x.astype(mx.float32) @ self.weight.astype(mx.float32).T).astype(x.dtype)


# ─── Qwen2 LM Decoder ────────────────────────────────────────────────────────

class Qwen2Layer:
    def __init__(self, weights, prefix, dtype=mx.float16, quantize=False):
        p = prefix
        load = lambda key: to_mx_q(weights[p + key], dtype, quantize=quantize)
        self.input_ln_w = to_mx(weights[p + "input_layernorm.weight"], dtype)
        self.post_ln_w = to_mx(weights[p + "post_attention_layernorm.weight"], dtype)
        self.q_proj = load("self_attn.q_proj.weight")
        self.q_bias = to_mx(weights[p + "self_attn.q_proj.bias"], dtype)
        self.k_proj = load("self_attn.k_proj.weight")
        self.k_bias = to_mx(weights[p + "self_attn.k_proj.bias"], dtype)
        self.v_proj = load("self_attn.v_proj.weight")
        self.v_bias = to_mx(weights[p + "self_attn.v_proj.bias"], dtype)
        self.o_proj = load("self_attn.o_proj.weight")
        self.gate_proj = load("mlp.gate_proj.weight")
        self.up_proj = load("mlp.up_proj.weight")
        self.down_proj = load("mlp.down_proj.weight")


class Qwen2Decoder:
    def __init__(self, weights, prefix, num_layers, num_q_heads, num_kv_heads,
                 head_dim, hidden_size, rms_norm_eps, has_norm=True, dtype=mx.float16):
        self.layers = []
        for i in range(num_layers):
            self.layers.append(Qwen2Layer(weights, f"{prefix}layers.{i}.", dtype))
        self.norm_w = to_mx(weights[f"{prefix}norm.weight"], dtype) if has_norm else None
        self.has_norm = has_norm
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.gqa_repeat = num_q_heads // num_kv_heads
        self.rms_norm_eps = rms_norm_eps

    def __call__(self, hidden, cos, sin):
        """Single-token decode (no KV cache — measures raw compute only)."""
        B, Q, H = hidden.shape

        for layer in self.layers:
            residual = hidden
            h = rms_norm(hidden, self.rms_norm_eps, layer.input_ln_w)

            q = (qmm(h, layer.q_proj) + layer.q_bias).reshape(B, Q, self.num_q_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = (qmm(h, layer.k_proj) + layer.k_bias).reshape(B, Q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = (qmm(h, layer.v_proj) + layer.v_bias).reshape(B, Q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            k = mx.repeat(k, self.gqa_repeat, axis=1)
            v = mx.repeat(v, self.gqa_repeat, axis=1)

            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(0, 1, 3, 2)) * scale
            attn = mx.softmax(attn, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, Q, H)
            hidden = residual + qmm(out, layer.o_proj)

            residual = hidden
            h = rms_norm(hidden, self.rms_norm_eps, layer.post_ln_w)
            gate = nn.silu(qmm(h, layer.gate_proj))
            up = qmm(h, layer.up_proj)
            hidden = residual + qmm(gate * up, layer.down_proj)

        if self.has_norm:
            hidden = rms_norm(hidden, self.rms_norm_eps, self.norm_w)
        return hidden
