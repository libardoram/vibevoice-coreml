# VibeVoice-1.5B CoreML Conversion

## Model Overview

**VibeVoice-1.5B** is a multi-speaker text-to-speech model from Microsoft that
generates expressive, long-form conversational audio (podcasts, dialogues) with
up to 4 distinct speakers and voice cloning from audio prompts.

- **Source**: [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)
- **License**: MIT
- **Total parameters**: ~3B (LLM 1.5B + tokenizers ~680M + diffusion ~123M)
- **Languages**: English, Chinese
- **Speakers**: Up to 4 distinct speakers with voice cloning
- **Max generation**: ~90 minutes
- **Context window**: 64K tokens (trained with curriculum: 4K → 16K → 32K → 64K)
- **Sample rate**: 24 kHz

## Architecture

Unlike the streaming 0.5B model, this uses a **single unified Qwen2.5-1.5B** LM
with autoregressive token generation. The model predicts control tokens
(`speech_start`, `speech_diffusion`, `speech_end`, `eos`) that direct the pipeline
between text and speech generation phases. Voice cloning is supported through
acoustic+semantic tokenizer encoding of reference audio.

```
Text + voice prompt
    │
    ▼
┌─────────────────────────────┐
│  Acoustic Tokenizer Encoder │  σ-VAE encoder (~340M), vae_dim=64
│  + Semantic Tokenizer Enc.  │  Semantic encoder (~340M), sem_dim=128
│  (voice cloning only)       │  Audio → latent features for speaker embedding
└──────────────┬──────────────┘
               │ acoustic_features + semantic_features
               ▼
┌─────────────────────────────┐
│  Connectors                 │  acoustic: Linear(64→1536) + RMSNorm + Linear
│                             │  semantic: Linear(128→1536) + RMSNorm + Linear
│                             │  Combined: acoustic_embed + semantic_embed
└──────────────┬──────────────┘
               │ speech embeddings inserted at speech_input_mask positions
               ▼
┌─────────────────────────────┐
│  Qwen2.5-1.5B LM           │  28 layers, hidden=1536
│  (unified decoder)          │  12 Q-heads, 2 KV-heads, head_dim=128
│                             │  intermediate=8960, SiLU activation
│                             │  GQA repeat factor: 6
│                             │  RoPE θ=1,000,000, RMSNorm ε=1e-6
│                             │  vocab=151,936 (Qwen2.5 tokenizer)
│                             │  tie_word_embeddings=true
└──────────────┬──────────────┘
               │
          ┌────┴────┐
          ▼         ▼
┌──────────────┐  ┌──────────────────────┐
│  LM Head     │  │  Diffusion Head      │
│  Linear →    │  │  4 HeadLayers (~123M) │
│  logits      │  │  hidden=1536          │
│  (151936)    │  │  latent=64            │
│              │  │  20 DDPM steps        │
│  Predicts:   │  │  cosine schedule      │
│  speech_start│  │  v_prediction         │
│  speech_diff │  │  CFG guidance         │
│  speech_end  │  └──────────┬───────────┘
│  eos         │             │ speech latent (1, 64)
└──────────────┘             ▼
                  ┌──────────────────────┐
                  │  VAE Decoder (~340M)  │
                  │  latent → 24kHz audio │
                  │  3200 samples/frame   │
                  └──────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Semantic Encoder     │  Encode generated audio chunk
                  │  (feedback loop)      │  → semantic features
                  │                       │  Combined with acoustic latent
                  │                       │  → fed back as next LM input
                  └──────────────────────┘
```

### Decoder Config (from config.json)

```json
{
  "model_type": "qwen2",
  "hidden_size": 1536,
  "num_hidden_layers": 28,
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "intermediate_size": 8960,
  "max_position_embeddings": 65536,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": true,
  "hidden_act": "silu"
}
```

### Diffusion Head Config

```json
{
  "hidden_size": 1536,
  "latent_size": 64,
  "head_layers": 4,
  "head_ffn_ratio": 3.0,
  "ddpm_num_steps": 1000,
  "ddpm_num_inference_steps": 20,
  "ddpm_beta_schedule": "cosine",
  "prediction_type": "v_prediction",
  "ddpm_batch_mul": 4
}
```

### Tokenizer Configs

**Acoustic** (σ-VAE, vae_dim=64, fix_std=0.5, gaussian sampling):
- Encoder/decoder: 7 stages, depths=[3,3,3,3,3,3,8], ratios=[8,5,5,4,2,2]
- n_filters=32, depthwise_conv mixer, causal, RMSNorm
- Total downsampling: 8×5×5×4×2×2 = 3200x → 7.5 Hz frame rate at 24kHz

**Semantic** (encoder only, vae_dim=128, no VAE sampling):
- Same encoder architecture as acoustic
- Trained with ASR proxy task

### Generation Loop (non-streaming)

```
1. Prefill: Encode voice prompt through tokenizers + connectors → insert into text
2. Autoregressive loop:
   a. Forward through LM → logits
   b. Sample next token (constrained to: speech_start, speech_diffusion, speech_end, eos)
   c. If speech_diffusion:
      - Get positive condition from LM hidden state
      - Get negative condition from negative prompt LM
      - Run 20 DDPM steps with CFG → speech latent
      - Decode latent through VAE → audio chunk
      - Encode audio through semantic tokenizer → semantic features
      - Combine: acoustic_connector(latent) + semantic_connector(semantic) → next input
   d. If speech_end: clear tokenizer streaming caches
   e. If eos: stop
```

## CoreML Export

### Scripts

| Script | Purpose |
|--------|---------|
| `convert_coreml.py` | Exports all non-LLM components |
| `convert_stateful_lm.py` | Exports 28-layer Qwen2 with stateful KV cache |

### Produced Artifacts

| File | Description | Size (est.) |
|------|-------------|-------------|
| `lm_decoder_stateful.mlpackage` | 28-layer Qwen2.5-1.5B, 56 KV buffers | ~3GB |
| `diffusion_head.mlpackage` | Single DDPM step | ~250MB |
| `vae_decoder.mlpackage` | σ-VAE decoder | ~680MB |
| `vae_encoder.mlpackage` | σ-VAE encoder (voice cloning) | ~680MB |
| `semantic_encoder.mlpackage` | Semantic tokenizer encoder | ~680MB |
| `acoustic_connector.mlpackage` | Latent→embedding | ~5MB |
| `semantic_connector.mlpackage` | Semantic→embedding | ~5MB |
| `lm_head.mlpackage` | Hidden→logits (151936) | ~900MB |
| `embed_tokens.npy` | Token embeddings (151936, 1536) | ~900MB |

### Usage

```bash
cd models/tts/vibevoice-1.5b/coreml
uv sync
uv run python convert_coreml.py --output-dir ./build/vibevoice-1.5b
uv run python convert_stateful_lm.py --output-dir ./build/vibevoice-1.5b
```

## Also Supports VibeVoice-7B

The same scripts work for the 7B model by passing `--model-id`:

```bash
uv run python convert_coreml.py \
    --model-id vibevoice/VibeVoice-7B \
    --output-dir ./build/vibevoice-7b

uv run python convert_stateful_lm.py \
    --model-id vibevoice/VibeVoice-7B \
    --output-dir ./build/vibevoice-7b \
    --max-seq-len 32768
```

**7B differences**: hidden=3584, 28 Q-heads, 4 KV-heads, intermediate=18944,
vocab=152064, diffusion head ~600M. Same architecture, just wider.
