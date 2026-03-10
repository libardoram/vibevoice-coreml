#!/usr/bin/env python3
"""Verify VibeVoice-Realtime-0.5B: correctness + performance.

Compares PyTorch (reference) against CoreML and/or MLX backends.
Reports max/mean absolute differences and latency for each component.

Usage:
    uv run python verify/verify.py                 # PT + CoreML + MLX (all)
    uv run python verify/verify.py --coreml        # PT + CoreML
    uv run python verify/verify.py --mlx           # PT + MLX
    uv run python verify/verify.py --coreml --mlx  # PT + CoreML + MLX
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
logging.getLogger("coremltools").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import numpy as np
import torch

from diffusion import ALPHAS_CUMPROD, VAE_DIM, ddpm_step_v
from rope import ROPE_THETA, compute_rope_np
from verify_common import (
    benchmark, benchmark_mlx, compare, print_latency_row,
    print_summary_table, pt_qwen2_forward,
)

# Architecture constants
BUILD_DIR = Path(__file__).resolve().parent.parent / "build/vibevoice-realtime-0.5b"
HIDDEN_SIZE = 896
NUM_Q_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
RMS_NORM_EPS = 1e-6
BASE_LM_LAYERS = 4
TTS_LM_LAYERS = 20

WARMUP = 10
ITERS = 100


def main():
    parser = argparse.ArgumentParser(description="Verify VibeVoice-Realtime-0.5B")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    parser.add_argument("--coreml", action="store_true", help="Verify CoreML backend")
    parser.add_argument("--mlx", action="store_true", help="Verify MLX backend")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    # Default to all backends if none specified
    if not args.coreml and not args.mlx:
        args.coreml = args.mlx = True

    build_dir = args.build_dir
    use_coreml = args.coreml
    use_mlx = args.mlx
    warmup = args.warmup
    iters = args.iters
    backends = (["coreml"] if use_coreml else []) + (["mlx"] if use_mlx else [])

    # ─── Load PyTorch model ──────────────────────────────────────
    print("Loading PyTorch model...")
    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    pt_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B", torch_dtype=torch.float32,
    )
    pt_model.eval()

    # Load Qwen2 for LM reference
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    model_path = Path.home() / ".cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots"
    st_files = sorted(model_path.rglob("model*.safetensors"))
    weights = {}
    for f in st_files:
        weights.update(load_file(str(f)))

    base_qwen = Qwen2Model(Qwen2Config(
        hidden_size=HIDDEN_SIZE, intermediate_size=4864, num_hidden_layers=4,
        num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
        max_position_embeddings=8192, rms_norm_eps=1e-6, rope_theta=1e6, hidden_act="silu",
    )).eval()
    base_w = {k[len("model.language_model."):]: v.float()
              for k, v in weights.items() if k.startswith("model.language_model.")}
    base_qwen.load_state_dict(base_w, strict=False)

    tts_qwen = Qwen2Model(Qwen2Config(
        hidden_size=HIDDEN_SIZE, intermediate_size=4864, num_hidden_layers=20,
        num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
        max_position_embeddings=8192, rms_norm_eps=1e-6, rope_theta=1e6, hidden_act="silu",
    )).eval()
    tts_w = {k[len("model.tts_language_model."):]: v.float()
             for k, v in weights.items() if k.startswith("model.tts_language_model.")}
    tts_qwen.load_state_dict(tts_w, strict=False)

    # ─── Load CoreML models ──────────────────────────────────────
    if use_coreml:
        import coremltools as ct
        print("Loading CoreML models...")
        cml_diff = ct.models.MLModel(str(build_dir / "diffusion_head.mlpackage"))
        cml_vae = ct.models.MLModel(str(build_dir / "vae_decoder.mlpackage"))
        cml_eos = ct.models.MLModel(str(build_dir / "eos_classifier.mlpackage"))
        cml_conn = ct.models.MLModel(str(build_dir / "acoustic_connector.mlpackage"))
        cml_base = ct.models.MLModel(str(build_dir / "base_lm_stateful.mlpackage"))
        cml_tts = ct.models.MLModel(str(build_dir / "tts_lm_stateful.mlpackage"))

    # ─── Load MLX models ─────────────────────────────────────────
    if use_mlx:
        import mlx.core as mx
        from bench_mlx import (
            AcousticConnector as MlxAcousticConnector,
            DiffusionHead as MlxDiffusionHead,
            EOSClassifier as MlxEOSClassifier,
            VAEDecoder as MlxVAEDecoder,
            Qwen2Decoder as MlxQwen2Decoder,
            make_rope as mlx_make_rope,
            to_mx,
        )
        print("Loading MLX models...")
        dtype = mx.float16
        mlx_diff = MlxDiffusionHead(weights, dtype)
        mlx_vae = MlxVAEDecoder(weights, dtype)
        mlx_eos = MlxEOSClassifier(weights, dtype)
        mlx_conn = MlxAcousticConnector(weights, dtype)
        mlx_base = MlxQwen2Decoder(weights, "model.language_model.", 4,
                                    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
                                    RMS_NORM_EPS, has_norm=False, dtype=dtype)
        mlx_tts = MlxQwen2Decoder(weights, "model.tts_language_model.", 20,
                                   NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
                                   RMS_NORM_EPS, has_norm=True, dtype=dtype)

    all_pass = True
    perf = {}
    active = ["PT"] + [b.upper()[:6] for b in backends]

    print(f"\nModels loaded.")
    print("=" * 72)
    print(f"  CORRECTNESS & PERFORMANCE — backends: {', '.join(active)}")
    print("=" * 72)

    # ─── Diffusion Head (single step) ────────────────────────────
    np.random.seed(42)
    noisy_np = np.random.randn(1, VAE_DIM).astype(np.float32)
    cond_np = np.random.randn(1, HIDDEN_SIZE).astype(np.float32)
    t_np = np.array([500.0], dtype=np.float32)

    head_pt = pt_model.model.prediction_head
    with torch.no_grad():
        pt_out = head_pt(torch.from_numpy(noisy_np), torch.tensor([500.0]),
                         condition=torch.from_numpy(cond_np)).numpy()

    outputs = {"PT": pt_out}
    lat = {"PT": benchmark(lambda: head_pt(torch.from_numpy(noisy_np),
                           torch.tensor([500.0]), condition=torch.from_numpy(cond_np)),
                           warmup, iters)}
    if use_coreml:
        outputs["COREML"] = cml_diff.predict({"noisy_latent": noisy_np, "timestep": t_np,
                                               "condition": cond_np})["predicted_noise"]
        lat["COREML"] = benchmark(lambda: cml_diff.predict({"noisy_latent": noisy_np,
                                  "timestep": t_np, "condition": cond_np}), warmup, iters)
    if use_mlx:
        noisy_mx = mx.array(noisy_np).astype(dtype)
        cond_mx = mx.array(cond_np).astype(dtype)
        t_mx = mx.array(t_np).astype(dtype)
        out = mlx_diff(noisy_mx, t_mx, cond_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_diff(noisy_mx, t_mx, cond_mx), warmup, iters)

    print(f"\n--- Diffusion Head (single step) ---")
    all_pass &= compare("diff", outputs)
    print_latency_row("diff_step", lat)
    perf["diff_step"] = lat

    # ─── DDPM 20-step loop ───────────────────────────────────────
    alphas = ALPHAS_CUMPROD
    step_ratio = 1000 // 20
    timesteps = (np.arange(20) * step_ratio)[::-1].astype(np.int64).copy()

    np.random.seed(42)
    cond_ddpm_np = np.random.randn(1, HIDDEN_SIZE).astype(np.float32)
    noise_ddpm = np.random.randn(1, VAE_DIM).astype(np.float32)

    def _ddpm_loop_pt():
        s = noise_ddpm.copy()
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                pred = head_pt(torch.from_numpy(s), torch.tensor([float(t)]),
                               condition=torch.from_numpy(cond_ddpm_np)).numpy()
                at = float(alphas[int(t)])
                ap = float(alphas[int(timesteps[i + 1])]) if i < 19 else 1.0
                s = ddpm_step_v(s, pred, at, ap)
        return s

    outputs = {"PT": _ddpm_loop_pt()}
    lat = {"PT": benchmark(_ddpm_loop_pt, warmup=3, iters=20)}

    if use_coreml:
        def _ddpm_loop_cml():
            s = noise_ddpm.copy()
            for i, t in enumerate(timesteps):
                pred = cml_diff.predict({"noisy_latent": s,
                    "timestep": np.array([float(t)], dtype=np.float32),
                    "condition": cond_ddpm_np})["predicted_noise"]
                at = float(alphas[int(t)])
                ap = float(alphas[int(timesteps[i + 1])]) if i < 19 else 1.0
                s = ddpm_step_v(s, pred, at, ap)
            return s
        outputs["COREML"] = _ddpm_loop_cml()
        lat["COREML"] = benchmark(_ddpm_loop_cml, warmup=3, iters=20)

    if use_mlx:
        cond_ddpm_mx = mx.array(cond_ddpm_np).astype(dtype)
        def _ddpm_loop_mlx():
            s = mx.array(noise_ddpm).astype(dtype)
            for i, t in enumerate(timesteps):
                pred = mlx_diff(s, mx.array([float(t)]).astype(dtype), cond_ddpm_mx)
                mx.eval(pred)
                s_np = np.array(s).astype(np.float32)
                pred_np = np.array(pred).astype(np.float32)
                at = float(alphas[int(t)])
                ap = float(alphas[int(timesteps[i + 1])]) if i < 19 else 1.0
                s_np = ddpm_step_v(s_np, pred_np, at, ap)
                s = mx.array(s_np).astype(dtype)
            mx.eval(s)
            return s
        out = _ddpm_loop_mlx()
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(_ddpm_loop_mlx, warmup=3, iters=20)

    print(f"\n--- DDPM Loop (20 steps) ---")
    all_pass &= compare("ddpm", outputs, threshold=0.5)
    print_latency_row("ddpm_20", lat)
    perf["ddpm_20"] = lat

    # ─── VAE Decoder ─────────────────────────────────────────────
    np.random.seed(42)
    latent_np = np.random.randn(1, VAE_DIM, 1).astype(np.float32)

    decoder_pt = pt_model.model.acoustic_tokenizer.decoder
    with torch.no_grad():
        pt_vae = decoder_pt(torch.from_numpy(latent_np)).numpy()

    outputs = {"PT": pt_vae}
    lat = {"PT": benchmark(lambda: decoder_pt(torch.from_numpy(latent_np)), warmup, iters)}
    if use_coreml:
        outputs["COREML"] = cml_vae.predict({"latent": latent_np})["audio"]
        lat["COREML"] = benchmark(lambda: cml_vae.predict({"latent": latent_np}), warmup, iters)
    if use_mlx:
        latent_mx = mx.array(latent_np).astype(dtype)
        out = mlx_vae(latent_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_vae(latent_mx), warmup, iters)

    print(f"\n--- VAE Decoder ---")
    all_pass &= compare("vae", outputs, threshold=0.5)
    print_latency_row("vae", lat)
    perf["vae"] = lat

    # ─── EOS Classifier ──────────────────────────────────────────
    np.random.seed(42)
    hidden_np = np.random.randn(1, HIDDEN_SIZE).astype(np.float32)

    eos_pt = pt_model.tts_eos_classifier
    with torch.no_grad():
        pt_eos = torch.sigmoid(eos_pt(torch.from_numpy(hidden_np))).numpy()

    outputs = {"PT": pt_eos}
    lat = {"PT": benchmark(lambda: torch.sigmoid(eos_pt(torch.from_numpy(hidden_np))),
                           warmup, iters)}
    if use_coreml:
        outputs["COREML"] = cml_eos.predict({"hidden_state": hidden_np})["eos_probability"]
        lat["COREML"] = benchmark(lambda: cml_eos.predict({"hidden_state": hidden_np}),
                                  warmup, iters)
    if use_mlx:
        hidden_mx = mx.array(hidden_np).astype(dtype)
        out = mlx_eos(hidden_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_eos(hidden_mx), warmup, iters)

    print(f"\n--- EOS Classifier ---")
    all_pass &= compare("eos", outputs)
    print_latency_row("eos", lat)
    perf["eos"] = lat

    # ─── Acoustic Connector ──────────────────────────────────────
    np.random.seed(42)
    lat_np = np.random.randn(1, 1, VAE_DIM).astype(np.float32)

    conn_pt = pt_model.model.acoustic_connector
    with torch.no_grad():
        pt_conn = conn_pt(torch.from_numpy(lat_np)).numpy()

    outputs = {"PT": pt_conn}
    lat = {"PT": benchmark(lambda: conn_pt(torch.from_numpy(lat_np)), warmup, iters)}
    if use_coreml:
        outputs["COREML"] = cml_conn.predict({"speech_latent": lat_np})["embedding"]
        lat["COREML"] = benchmark(lambda: cml_conn.predict({"speech_latent": lat_np}),
                                  warmup, iters)
    if use_mlx:
        lat_mx = mx.array(lat_np).astype(dtype)
        out = mlx_conn(lat_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_conn(lat_mx), warmup, iters)

    print(f"\n--- Acoustic Connector ---")
    all_pass &= compare("connector", outputs)
    print_latency_row("connector", lat)
    perf["connector"] = lat

    # ─── Base LM (4 layers, single-token decode) ─────────────────
    np.random.seed(42)
    h_np = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)
    cos_np, sin_np = compute_rope_np(0, HEAD_DIM)

    pt_base_out = pt_qwen2_forward(base_qwen.layers, base_qwen.norm, False,
                                    h_np, cos_np, sin_np, HIDDEN_SIZE,
                                    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    outputs = {"PT": pt_base_out}
    lat = {"PT": benchmark(lambda: pt_qwen2_forward(
        base_qwen.layers, base_qwen.norm, False, h_np, cos_np, sin_np,
        HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM), warmup, iters)}

    if use_coreml:
        base_total_kv = BASE_LM_LAYERS * NUM_KV_HEADS
        base_inject = {
            "inject_mode": np.zeros((1,), dtype=np.float32),
            "inject_k": np.zeros((1, base_total_kv, 1, HEAD_DIM), dtype=np.float32),
            "inject_v": np.zeros((1, base_total_kv, 1, HEAD_DIM), dtype=np.float32),
        }
        state_base = cml_base.make_state()
        outputs["COREML"] = cml_base.predict({
            "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
            **base_inject,
        }, state=state_base)["output_hidden"]
        def _cml_base_fn():
            s = cml_base.make_state()
            cml_base.predict({
                "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
                "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
                **base_inject,
            }, state=s)
        lat["COREML"] = benchmark(_cml_base_fn, warmup, iters)

    if use_mlx:
        h_mx = mx.array(h_np).astype(dtype)
        cos_mx, sin_mx = mlx_make_rope(mx.array(0.0), HEAD_DIM, ROPE_THETA, dtype)
        out = mlx_base(h_mx, cos_mx, sin_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_base(h_mx, cos_mx, sin_mx), warmup, iters)

    print(f"\n--- Base LM (4 layers, decode) ---")
    all_pass &= compare("base_lm", outputs, threshold=0.01)
    print_latency_row("base_lm", lat)
    perf["base_lm"] = lat

    # ─── TTS LM (20 layers, single-token decode) ─────────────────
    pt_tts_out = pt_qwen2_forward(tts_qwen.layers, tts_qwen.norm, True,
                                   h_np, cos_np, sin_np, HIDDEN_SIZE,
                                   NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    outputs = {"PT": pt_tts_out}
    lat = {"PT": benchmark(lambda: pt_qwen2_forward(
        tts_qwen.layers, tts_qwen.norm, True, h_np, cos_np, sin_np,
        HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM), warmup, iters)}

    if use_coreml:
        tts_total_kv = TTS_LM_LAYERS * NUM_KV_HEADS
        tts_inject = {
            "inject_mode": np.zeros((1,), dtype=np.float32),
            "inject_k": np.zeros((1, tts_total_kv, 1, HEAD_DIM), dtype=np.float32),
            "inject_v": np.zeros((1, tts_total_kv, 1, HEAD_DIM), dtype=np.float32),
        }
        state_tts = cml_tts.make_state()
        outputs["COREML"] = cml_tts.predict({
            "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
            **tts_inject,
        }, state=state_tts)["output_hidden"]
        def _cml_tts_fn():
            s = cml_tts.make_state()
            cml_tts.predict({
                "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
                "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
                **tts_inject,
            }, state=s)
        lat["COREML"] = benchmark(_cml_tts_fn, warmup, iters)

    if use_mlx:
        out = mlx_tts(h_mx, cos_mx, sin_mx)
        mx.eval(out)
        outputs["MLX"] = np.array(out)
        lat["MLX"] = benchmark_mlx(lambda: mlx_tts(h_mx, cos_mx, sin_mx), warmup, iters)

    print(f"\n--- TTS LM (20 layers, decode) ---")
    all_pass &= compare("tts_lm", outputs, threshold=0.05)
    print_latency_row("tts_lm", lat)
    perf["tts_lm"] = lat

    # ─── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  CORRECTNESS: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 72}")

    all_backends = ["PT"] + [b.upper()[:6] for b in backends]
    print(f"\n{'=' * 72}")
    print(f"  PERFORMANCE (median ms, {iters} iters, {warmup} warmup)")
    print(f"{'=' * 72}")
    print_summary_table(perf, all_backends, warmup, iters)

    # Per-frame pipeline estimate
    frame_keys = ["tts_lm", "ddpm_20", "vae"]
    if all(k in perf for k in frame_keys):
        print(f"\n  --- Per-frame estimate ---")
        audio_ms = 3200 / 24000 * 1000
        for b in all_backends:
            total = sum(perf[k].get(b, {}).get("median_ms", 0) for k in frame_keys)
            if total > 0:
                print(f"  {b}: {total:.1f}ms / {audio_ms:.1f}ms audio = {audio_ms/total:.1f}x RT")

    print(f"\n  Note: LM benchmarks create fresh state each iter (first-token latency).")


if __name__ == "__main__":
    main()
