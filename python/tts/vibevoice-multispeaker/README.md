# VibeVoice Multi-Speaker CoreML Conversion

## Model Overview

**VibeVoice** multi-speaker TTS models from Microsoft generate expressive,
long-form conversational audio (podcasts, dialogues) with up to 4 distinct
speakers and voice cloning from audio prompts.

This directory supports both model sizes:

| | **VibeVoice-1.5B** | **VibeVoice-7B** |
|---|---|---|
| Source | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | [vibevoice/VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) |
| LLM | Qwen2.5-1.5B (h=1536, 12Q/2KV) | Qwen2.5-7B (h=3584, 28Q/4KV) |
| Vocab | 151,936 | 152,064 |
| Context | 64K tokens | 32K tokens |
| LM head | Tied (shared with embed_tokens) | Untied (separate weights) |
| Diffusion | ~123M params | ~600M params |

Both share: MIT license, 24kHz sample rate, 28 decoder layers, same tokenizer
architecture, same generation loop, up to 4 speakers.

## Architecture

```
Text + voice prompt
    │
    ▼
┌─────────────────────────────┐
│  Acoustic Tokenizer Encoder │  σ-VAE encoder, vae_dim=64
│  + Semantic Tokenizer Enc.  │  Semantic encoder, sem_dim=128
│  (voice cloning only)       │  Audio → latent features for speaker embedding
└──────────────┬──────────────┘
               │ acoustic_features + semantic_features
               ▼
┌─────────────────────────────┐
│  Connectors                 │  acoustic: Linear(64→H) + RMSNorm + Linear
│                             │  semantic: Linear(128→H) + RMSNorm + Linear
│                             │  Combined: acoustic_embed + semantic_embed
└──────────────┬──────────────┘
               │ speech embeddings inserted at speech_input_mask positions
               ▼
┌─────────────────────────────┐
│  Qwen2.5 LM (unified)      │  28 layers, GQA attention
│                             │  RoPE θ=1,000,000, RMSNorm, SiLU
└──────────────┬──────────────┘
               │
          ┌────┴────┐
          ▼         ▼
┌──────────────┐  ┌──────────────────────┐
│  LM Head     │  │  Diffusion Head      │
│  Linear →    │  │  4 HeadLayers        │
│  logits      │  │  DDPM v_prediction   │
│              │  │  cosine schedule     │
│  Predicts:   │  │  CFG guidance        │
│  speech_start│  └──────────┬───────────┘
│  speech_diff │             │ speech latent (1, 64)
│  speech_end  │             ▼
│  eos         │  ┌──────────────────────┐
└──────────────┘  │  VAE Decoder         │
                  │  latent → 24kHz audio│
                  │  3200 samples/frame  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Semantic Encoder     │  Encode generated audio chunk
                  │  (feedback loop)      │  → semantic features
                  │                       │  Combined with acoustic latent
                  │                       │  → fed back as next LM input
                  └──────────────────────┘
```

### Generation Loop

1. Prefill: encode voice prompt through tokenizers + connectors → insert into text
2. Autoregressive loop:
   - Forward through LM → logits
   - Sample next token (constrained to: speech_start, speech_diffusion, speech_end, eos)
   - If `speech_diffusion`: run diffusion → VAE decode → semantic encode → feedback
   - If `speech_end`: clear streaming caches
   - If `eos`: stop

## Quick Start

```bash
cd python/tts/vibevoice-multispeaker
uv sync

# Convert CoreML models (1.5B with INT8 + fused LM+head)
uv run python convert/convert_all.py --model-id microsoft/VibeVoice-1.5B --int8 --fuse-lm-head

# Convert CoreML models (7B with INT8 + fused LM+head)
uv run python convert/convert_all.py --model-id vibevoice/VibeVoice-7B --int8 --fuse-lm-head

# Run MLX backend (no CoreML conversion needed, downloads MLX weights from HF)
uv run python run/e2e_pipeline.py --model-id microsoft/VibeVoice-1.5B --mlx --int8 --solver dpm --diffusion-steps 10

# Run CoreML backend
uv run python run/e2e_pipeline.py --model-id microsoft/VibeVoice-1.5B --coreml --int8 --solver dpm --diffusion-steps 10

# Compare all backends
uv run python run/e2e_pipeline.py --model-id microsoft/VibeVoice-1.5B --pytorch --coreml --mlx --int8 --solver dpm --diffusion-steps 10
```

## Scripts

| Directory | Script | Purpose |
|-----------|--------|---------|
| `convert/` | `convert_all.py` | Full pipeline: convert all components + optimize |
| | `convert_coreml.py` | Export non-LLM components (diffusion, VAE, connectors) |
| | `convert_stateful_lm.py` | Export Qwen2.5 decoder with stateful KV cache |
| | `convert_streaming_semantic.py` | Export streaming semantic encoder with conv caches |
| | `traceable_modules.py` | PyTorch traceable wrappers for CoreML export |
| `verify/` | `verify.py` | Correctness + performance verification |
| | `verify_coreml.py` | CoreML backend verification |
| | `verify_mlx.py` | MLX backend verification |
| `run/` | `e2e_pipeline.py` | End-to-end text→audio benchmark (PyTorch/CoreML/MLX) |
| | `pipeline_common.py` | Shared constants, configs, metrics, tokenizer |
| | `pipeline_pytorch.py` | PyTorch backend |
| | `pipeline_coreml.py` | CoreML backend |
| | `pipeline_mlx.py` | MLX backend (via [vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx)) |

## Optimizations

`convert_all.py` supports optimization flags:

| Flag | Effect |
|------|--------|
| `--int8` | INT8 weight quantization (W8A16), ~50% size reduction |
| `--fuse-lm-head` | Fuse LM decoder + head into one model, saves 1 dispatch/token |

When optimizations are selected, intermediate models are cleaned up automatically.
Only the final optimized variant is kept:

| Flags | Final LM model |
|-------|----------------|
| (none) | `lm_decoder_stateful` + `lm_head` |
| `--int8` | `lm_decoder_stateful_int8` + `lm_head` |
| `--fuse-lm-head` | `lm_decoder_fused` |
| `--int8 --fuse-lm-head` | `lm_decoder_fused_int8` |

## Output Artifacts

| File | Description |
|------|-------------|
| `lm_decoder_*.mlpackage` | Qwen2.5 decoder with KV cache (variant depends on flags) |
| `diffusion_head.mlpackage` | Single DDPM denoising step |
| `vae_decoder_streaming.mlpackage` | σ-VAE decoder (streaming, with conv caches) |
| `vae_encoder.mlpackage` | σ-VAE encoder (voice cloning) |
| `semantic_encoder_streaming.mlpackage` | Streaming semantic encoder (conv caches) |
| `acoustic_connector.mlpackage` | Latent→embedding projection |
| `semantic_connector.mlpackage` | Semantic→embedding projection |
| `embed_tokens.npy` | Token embedding table |
