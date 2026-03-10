#!/usr/bin/env python3
"""Verify VibeVoice multi-speaker models: correctness + performance.

Compares PyTorch (reference) against CoreML and/or MLX backends.
Reports max/mean absolute differences and latency for each component.

Usage:
    uv run python verify/verify.py --model-id microsoft/VibeVoice-1.5B
    uv run python verify/verify.py --model-id vibevoice/VibeVoice-7B --coreml
    uv run python verify/verify.py --model-id microsoft/VibeVoice-1.5B --mlx
"""

import argparse
import logging
import math
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
import torch.nn.functional as F

from diffusion import ALPHAS_CUMPROD, VAE_DIM, ddpm_step_v
from rope import compute_rope_np
from verify_common import (
    benchmark, compare, print_latency_row,
    print_summary_table, pt_qwen2_forward,
)

# Architecture constants (set by configure())
SEMANTIC_DIM = 128
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
SAMPLE_RATE = 24000

MODEL_CONFIGS = {
    "microsoft/VibeVoice-1.5B": {
        "hidden_size": 1536, "num_q_heads": 12, "num_kv_heads": 2,
        "num_layers": 28, "vocab_size": 151936, "intermediate_size": 8960,
        "build_dir": "build/vibevoice-1.5b",
    },
    "vibevoice/VibeVoice-7B": {
        "hidden_size": 3584, "num_q_heads": 28, "num_kv_heads": 4,
        "num_layers": 28, "vocab_size": 152064, "intermediate_size": 18944,
        "build_dir": "build/vibevoice-7b",
    },
}

WARMUP = 10
ITERS = 100


def main():
    parser = argparse.ArgumentParser(description="Verify VibeVoice multi-speaker models")
    parser.add_argument("--model-id", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model to verify")
    parser.add_argument("--coreml", action="store_true", help="Verify CoreML backend")
    parser.add_argument("--mlx", action="store_true", help="Verify MLX backend")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    # Default to all backends if none specified
    if not args.coreml and not args.mlx:
        args.coreml = args.mlx = True

    # Configure from model ID
    cfg = MODEL_CONFIGS[args.model_id]
    HIDDEN_SIZE = cfg["hidden_size"]
    NUM_Q_HEADS = cfg["num_q_heads"]
    NUM_KV_HEADS = cfg["num_kv_heads"]
    NUM_LAYERS = cfg["num_layers"]
    VOCAB_SIZE = cfg["vocab_size"]
    INTERMEDIATE_SIZE = cfg["intermediate_size"]
    build_dir = Path(__file__).resolve().parent.parent / cfg["build_dir"]

    use_coreml = args.coreml
    use_mlx = args.mlx
    backends = (["coreml"] if use_coreml else []) + (["mlx"] if use_mlx else [])
    warmup = args.warmup
    iters = args.iters

    print(f"Model: {args.model_id}")
    print(f"Config: {NUM_LAYERS}L, h={HIDDEN_SIZE}, {NUM_Q_HEADS}Q/{NUM_KV_HEADS}KV")

    # ─── Load PyTorch model ──────────────────────────────────────
    print("Loading PyTorch model...")
    import gc
    import transformers
    transformers.logging.set_verbosity_error()
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    pt_model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32,
    )
    pt_model.eval()

    # Use the internal language model layers + norm for LM reference (avoids loading a second copy)
    qwen_layers = pt_model.model.language_model.layers
    qwen_norm = pt_model.model.language_model.norm
    has_norm = True

    # ─── Load backend models ─────────────────────────────────────
    cml = None
    mlx_m = None
    if use_coreml:
        import verify_coreml
        print("Loading CoreML models...")
        cml = verify_coreml.load_models(build_dir)
    if use_mlx:
        from safetensors.torch import load_file
        cache_name = args.model_id.replace("/", "--")
        model_path = Path.home() / f".cache/huggingface/hub/models--{cache_name}/snapshots"
        st_files = sorted(model_path.rglob("model*.safetensors"))
        weights = {}
        for f in st_files:
            weights.update(load_file(str(f)))
        import verify_mlx
        print("Loading MLX models...")
        mlx_m = verify_mlx.load_models(
            weights, num_layers=NUM_LAYERS, num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            hidden_size=HIDDEN_SIZE, rms_norm_eps=RMS_NORM_EPS)
        del weights
        gc.collect()

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
        o, l = verify_coreml.test_diffusion(cml, noisy_np, t_np, cond_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_diffusion(mlx_m, noisy_np, t_np, cond_np, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

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
        o, l = verify_coreml.test_ddpm_loop(cml, noise_ddpm, cond_ddpm_np, timesteps, alphas, ddpm_step_v, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_ddpm_loop(mlx_m, noise_ddpm, cond_ddpm_np, timesteps, alphas, ddpm_step_v, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

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
        o, l = verify_coreml.test_vae(cml, latent_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_vae(mlx_m, latent_np, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

    print(f"\n--- VAE Decoder ---")
    all_pass &= compare("vae", outputs, threshold=0.5)
    print_latency_row("vae", lat)
    perf["vae"] = lat

    # ─── VAE Encoder (CoreML-only, optional) ─────────────────────
    if use_coreml:
        np.random.seed(42)
        audio_np = np.random.randn(1, 1, SAMPLE_RATE * 10).astype(np.float32)
        cml_enc = verify_coreml.test_vae_encoder(cml, audio_np)
        if cml_enc is not None:
            with torch.no_grad():
                pt_enc = pt_model.model.acoustic_tokenizer.encoder(torch.from_numpy(audio_np)).numpy()
            print(f"\n--- VAE Encoder ---")
            all_pass &= compare("vae_enc", {"PT": pt_enc, "COREML": cml_enc}, threshold=0.5)

    # ─── Semantic Encoder (CoreML-only, optional) ────────────────
    # Note: CoreML semantic encoder is streaming (causal, 3200-sample chunks)
    # while PT reference is non-streaming, so they diverge on initial frames.
    # Streaming correctness is validated during conversion (max diff ~2.7e-05).
    # We verify the model loads and runs without error.
    if use_coreml:
        np.random.seed(42)
        audio_sem = np.random.randn(1, 1, SAMPLE_RATE * 10).astype(np.float32)
        cml_sem = verify_coreml.test_semantic_encoder(cml, audio_sem)
        if cml_sem is not None:
            print(f"\n--- Semantic Encoder (streaming) ---")
            print(f"  CoreML output shape: {cml_sem.shape}  OK")

    # ─── Acoustic Connector ──────────────────────────────────────
    np.random.seed(42)
    lat_np = np.random.randn(1, 1, VAE_DIM).astype(np.float32)

    conn_pt = pt_model.model.acoustic_connector
    with torch.no_grad():
        pt_conn = conn_pt(torch.from_numpy(lat_np)).numpy()

    outputs = {"PT": pt_conn}
    lat = {"PT": benchmark(lambda: conn_pt(torch.from_numpy(lat_np)), warmup, iters)}
    if use_coreml:
        o, l = verify_coreml.test_acoustic_connector(cml, lat_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_acoustic_connector(mlx_m, lat_np, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

    print(f"\n--- Acoustic Connector ---")
    all_pass &= compare("ac_conn", outputs)
    print_latency_row("ac_connector", lat)
    perf["ac_connector"] = lat

    # ─── Semantic Connector ──────────────────────────────────────
    np.random.seed(42)
    feat_np = np.random.randn(1, 1, SEMANTIC_DIM).astype(np.float32)

    sem_pt = pt_model.model.semantic_connector
    with torch.no_grad():
        pt_sem = sem_pt(torch.from_numpy(feat_np)).numpy()

    outputs = {"PT": pt_sem}
    lat = {"PT": benchmark(lambda: sem_pt(torch.from_numpy(feat_np)), warmup, iters)}
    if use_coreml:
        o, l = verify_coreml.test_semantic_connector(cml, feat_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_semantic_connector(mlx_m, feat_np, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

    print(f"\n--- Semantic Connector ---")
    all_pass &= compare("sem_conn", outputs)
    print_latency_row("sem_connector", lat)
    perf["sem_connector"] = lat

    # ─── LM Head ─────────────────────────────────────────────────
    np.random.seed(42)
    hidden_np = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)

    with torch.no_grad():
        pt_lm_head = pt_model.lm_head(torch.from_numpy(hidden_np)).numpy()

    outputs = {"PT": pt_lm_head}
    lat = {"PT": benchmark(lambda: pt_model.lm_head(torch.from_numpy(hidden_np)), warmup, iters)}
    if use_coreml:
        o, l = verify_coreml.test_lm_head(cml, hidden_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_lm_head(mlx_m, hidden_np, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

    print(f"\n--- LM Head ---")
    all_pass &= compare("lm_head", outputs, threshold=0.5)
    print_latency_row("lm_head", lat)
    perf["lm_head"] = lat

    # ─── LM Decoder (28 layers, single-token decode) ─────────────
    np.random.seed(42)
    h_np = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)
    cos_np, sin_np = compute_rope_np(0, HEAD_DIM)

    pt_lm_out = pt_qwen2_forward(qwen_layers, qwen_norm, has_norm, h_np, cos_np, sin_np,
                                  HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    outputs = {"PT": pt_lm_out}
    lat = {"PT": benchmark(lambda: pt_qwen2_forward(
        qwen_layers, qwen_norm, has_norm, h_np, cos_np, sin_np,
        HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM), warmup, iters)}

    if use_coreml:
        o, l = verify_coreml.test_lm_decoder(cml, h_np, cos_np, sin_np, warmup, iters)
        outputs["COREML"] = o; lat["COREML"] = l
    if use_mlx:
        o, l = verify_mlx.test_lm_decoder(mlx_m, h_np, cos_np, sin_np, HEAD_DIM, warmup, iters)
        outputs["MLX"] = o; lat["MLX"] = l

    print(f"\n--- LM Decoder (28 layers, decode) ---")
    all_pass &= compare("lm_decoder", outputs, threshold=0.1)
    print_latency_row("lm_decoder", lat)
    perf["lm_decoder"] = lat

    # ─── Voice Cloning Pipeline (CoreML-only, optional) ──────────
    if use_coreml:
        np.random.seed(123)
        ref_audio = np.random.randn(1, 1, SAMPLE_RATE * 10).astype(np.float32)
        cml_ac_emb, cml_sem_emb = verify_coreml.test_voice_cloning(
            cml, ref_audio, cml["ac_conn"], cml["sem_conn"], SAMPLE_RATE)

        if cml_ac_emb is not None:
            ref_audio_t = torch.from_numpy(ref_audio)
            with torch.no_grad():
                pt_vae_lat = pt_model.model.acoustic_tokenizer.encoder(ref_audio_t)
                pt_ac_emb = pt_model.model.acoustic_connector(pt_vae_lat.permute(0, 2, 1))
                pt_sem_feat = pt_model.model.semantic_tokenizer.encoder(ref_audio_t)
                pt_sem_emb = pt_model.model.semantic_connector(pt_sem_feat.permute(0, 2, 1))

            print(f"\n--- Voice Cloning Pipeline ---")
            print("  (ref audio -> encoders -> connectors -> embeddings)")
            print("  Acoustic path:")
            all_pass &= compare("vc_acoustic", {"PT": pt_ac_emb.numpy(), "COREML": cml_ac_emb},
                                threshold=1.0)
            print("  Semantic path:")
            all_pass &= compare("vc_semantic", {"PT": pt_sem_emb.numpy(), "COREML": cml_sem_emb},
                                threshold=1.0)

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
    frame_keys = ["lm_decoder", "ddpm_20", "vae"]
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
