# VibeVoice-ASR CoreML

CoreML export for [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) speech recognition.

## Architecture

- **Acoustic encoder**: σ-VAE (24kHz audio → dim=64 features, 7.5 Hz)
- **Semantic encoder**: σ-VAE (24kHz audio → dim=128 features, 7.5 Hz)
- **Projectors**: Linear(dim→3584) + RMSNorm + Linear(3584→3584), summed
- **LM backbone**: Qwen2-7B (28 layers, h=3584, 28Q/4KV heads)
- **LM head**: Linear(3584→152064)

## Usage

```bash
# Install dependencies
uv sync

# Convert all components to CoreML
uv run python convert/convert_all.py

# With optimizations
uv run python convert/convert_all.py --int8 --fuse-lm-head --fuse-projectors

# Run end-to-end pipeline
uv run python run/e2e_pipeline.py --audio test.wav
uv run python run/e2e_pipeline.py --audio test.wav --coreml --int8 --fused-lm-head
```

## Optimizations

- **Batch prefill**: Feed all prompt embeddings (audio + text) in one LM call instead of token-by-token. Critical for long audio (5min = ~1500 tokens).
- **Fused projector**: Single model for acoustic+semantic projection (saves one CoreML dispatch per chunk).
- **INT8 quantization**: W8A16 per-block (block_size=32) for LM decoder. ~50% size reduction.
- **Fused LM+head**: Combined decoder + output projection, saves one dispatch per generated token.

## Components

| Component | Input | Output |
|-----------|-------|--------|
| acoustic_encoder | (1, 1, 1440000) audio | (1, T, 64) features |
| semantic_encoder | (1, 1, 1440000) audio | (1, T, 128) features |
| fused_projector | (1, T, 64) + (1, T, 128) | (1, T, 3584) embedding |
| lm_decoder_stateful | (1, Q, 3584) hidden | (1, Q, 3584) output |
| lm_head | (1, Q, 3584) hidden | (1, Q, 152064) logits |
| embed_tokens.npy | token_id | (3584,) embedding |
