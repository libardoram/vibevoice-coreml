# VibeVoice-Realtime-0.5B CoreML Conversion

## Model Overview

**VibeVoice-Realtime-0.5B** is a streaming text-to-speech model from Microsoft
designed for real-time audio generation with ~300ms first-audible latency. It
processes text incrementally in 5-token windows, generating 6 speech frames per
window through an interleaved encoding/generation pipeline.

- **Source**: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- **License**: MIT
- **Languages**: English (German, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish exploratory/unsupported)
- **Speakers**: Single speaker (pre-computed voice embeddings: Carter, Davis, Emma, Frank, Grace, Mike, Samuel)
- **Max generation**: ~10 minutes
- **Sample rate**: 24 kHz

## Architecture

The model splits a Qwen2.5-0.5B transformer into two sequential LMs, each with
independent KV caches. A diffusion head generates acoustic latents which a
σ-VAE decoder converts to audio.

```
Text tokens (5 at a time)
    │
    ▼
┌─────────────────────────┐
│  Base LM (4 layers)     │  Qwen2.5, hidden=896, 14Q/2KV heads, head_dim=64
│  No final norm          │  KV cache: 8 state buffers (4 layers × K+V)
└────────────┬────────────┘
             │ hidden states
             ▼
┌─────────────────────────┐
│  TTS LM (20 layers)     │  Same arch, separate weights + KV cache
│  + tts_input_types emb  │  40 state buffers (20 layers × K+V)
│  + RMSNorm final        │  Adds type embedding: text(1) or speech(0)
└────────────┬────────────┘
             │ hidden states
             ▼
┌─────────────────────────┐
│  EOS Classifier         │  Linear(896→896) + ReLU + Linear(896→1) + sigmoid
│                         │  Stops generation when prob > 0.5
└─────────────────────────┘
             │ condition (hidden_state[:, -1, :])
             ▼
┌─────────────────────────┐
│  Diffusion Head         │  4 HeadLayers + FinalLayer (~40M params)
│  DDPM v_prediction      │  20 denoising steps, cosine schedule
│  Classifier-Free        │  Positive + negative conditioning
│  Guidance (CFG)         │  noisy(1,64) + timestep + condition(1,896) → noise(1,64)
└────────────┬────────────┘
             │ speech latent (1, 64)
             ▼
┌─────────────────────────┐
│  Acoustic Connector     │  Linear(64→896) + RMSNorm + Linear(896→896)
│                         │  Projects latent to LM embedding space
└─────────────────────────┘  (fed back into TTS LM as next input)
             │
             ▼
┌─────────────────────────┐
│  VAE Decoder (~340M)    │  σ-VAE, 7 stages, ratios=[8,5,5,4,2,2]
│  Causal convolutions    │  depthwise_conv mixer, RMSNorm
│                         │  latent(1,1,64) → audio(1,1,3200) = 133ms @ 24kHz
└─────────────────────────┘
```

### Component Details

| Component | Parameters | Input | Output |
|-----------|-----------|-------|--------|
| Base LM | ~100M | token embeddings (1, Q, 896) | hidden states (1, Q, 896) |
| TTS LM | ~400M | hidden states + type emb (1, Q, 896) | hidden states (1, Q, 896) |
| EOS Classifier | ~1.6M | hidden (1, 896) | probability (1, 1) |
| Diffusion Head | ~40M | noisy (1, 64) + t (1,) + cond (1, 896) | noise (1, 64) |
| Acoustic Connector | ~1.6M | latent (1, 1, 64) | embedding (1, 1, 896) |
| VAE Decoder | ~340M | latent (1, 1, 64) | audio (1, 1, 3200) |

### Streaming Generation Loop

```
1. Prefill: Process voice prompt through both LMs (build KV cache)
2. For each text window (5 tokens):
   a. Feed tokens through Base LM → get hidden states
   b. Feed hidden states + type_emb(text=1) through TTS LM
   c. For each of 6 speech frames:
      i.   Get positive condition from TTS LM hidden state
      ii.  Get negative condition from negative prompt path
      iii. Run 20 DDPM steps: noise → latent (with CFG)
      iv.  Scale latent, decode through VAE → audio chunk
      v.   Stream audio chunk
      vi.  Project latent through connector → next TTS LM input
      vii. Feed with type_emb(speech=0) through TTS LM
      viii. Check EOS classifier → stop if > 0.5
   d. Advance to next text window
```

## CoreML Export

### Scripts

| Script | Purpose |
|--------|---------|
| `convert_coreml.py` | Exports diffusion head, VAE decoder, EOS classifier, acoustic connector |
| `convert_stateful_lm.py` | Exports Base LM (4L) and TTS LM (20L) with `ct.StateType` KV cache |

### Produced Artifacts

| File | Description | Size (est.) |
|------|-------------|-------------|
| `base_lm_stateful.mlpackage` | 4-layer Qwen2, 8 KV state buffers | ~200MB |
| `tts_lm_stateful.mlpackage` | 20-layer Qwen2, 40 KV state buffers | ~800MB |
| `diffusion_head.mlpackage` | Single DDPM denoising step | ~80MB |
| `vae_decoder.mlpackage` | σ-VAE decoder (convolutional) | ~680MB |
| `eos_classifier.mlpackage` | Binary end-of-speech | ~3MB |
| `acoustic_connector.mlpackage` | Latent→embedding projection | ~3MB |
| `embed_tokens.npy` | Token embedding table (151936, 896) | ~520MB |
| `tts_input_types.npy` | Type embeddings (2, 896) | ~7KB |
| `speech_scaling_factor.npy` | Normalization constant | ~4B |
| `speech_bias_factor.npy` | Normalization constant | ~4B |

### Usage

```bash
cd models/tts/vibevoice-realtime-0.5b/coreml
uv sync
uv run python convert_coreml.py --output-dir ./build/vibevoice-realtime-0.5b
uv run python convert_stateful_lm.py --output-dir ./build/vibevoice-realtime-0.5b
```

### CoreML Runtime Notes

- **Stateful models** require iOS 18+ / macOS 15+ (for `ct.StateType`)
- **KV cache** stored as fp16 GPU-resident buffers — no marshaling overhead
- **Variable sequence length** via `RangeDim` — same model for prefill and decode
- **Compute units**: `CPU_AND_GPU` for LLM backbones; `ALL` for other components
- **Diffusion loop**: 20 forward passes orchestrated externally (Swift/Python)
- **VAE decoder**: Fully convolutional — excellent candidate for GPU or ANE
