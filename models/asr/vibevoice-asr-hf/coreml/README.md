# VibeVoice-ASR-HF CoreML Conversion

## Model Overview

**VibeVoice-ASR-HF** is a unified speech-to-text model from Microsoft that processes
up to 60 minutes of continuous audio in a single pass, generating structured
transcriptions with speaker identification, timestamps, and content. It is the
HuggingFace Transformers-native variant of the original VibeVoice-ASR.

- **Source**: [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
- **License**: MIT
- **Total parameters**: 8B (LLM ~7B + tokenizers ~680M)
- **Languages**: 50+ languages with code-switching support
- **Max audio**: 60 minutes in a single pass
- **Sample rate**: 24 kHz input
- **Framework**: `transformers>=5.3.0`

### Comparison: ASR-HF vs ASR

| | VibeVoice-ASR-HF | VibeVoice-ASR |
|---|---|---|
| **Params** | 8B | 9B |
| **Framework** | HuggingFace Transformers (native) | Custom (vLLM plugin) |
| **Dependencies** | `transformers>=5.3.0` | Custom loader |
| **Recommended for CoreML** | Yes — clean model loading | No — non-standard format |

## Architecture

The ASR model encodes audio through two parallel tokenizer branches (acoustic +
semantic), projects them into the LLM embedding space, inserts them at
`acoustic_input_mask` positions in the text prompt, and autoregressively generates
structured transcription output.

```
Audio (24kHz, up to 60 minutes)
    │
    ├──────────────────────────┐
    ▼                          ▼
┌────────────────────┐  ┌────────────────────┐
│ Acoustic Tokenizer │  │ Semantic Tokenizer │
│ Encoder (~340M)    │  │ Encoder (~340M)    │
│ σ-VAE, vae_dim=64  │  │ sem_dim=128        │
│ 7 stages, causal   │  │ No VAE sampling    │
│ RMSNorm, depthwise │  │ Same conv arch     │
│                    │  │                    │
│ Streaming: 60s     │  │ Streaming: 60s     │
│ segments with      │  │ segments with      │
│ conv state cache   │  │ conv state cache   │
└────────┬───────────┘  └────────┬───────────┘
         │ (B, T, 64)           │ (B, T, 128)
         ▼                      ▼
┌────────────────────┐  ┌────────────────────┐
│ Acoustic Connector │  │ Semantic Connector │
│ Linear(64→3584)    │  │ Linear(128→3584)   │
│ + RMSNorm          │  │ + RMSNorm          │
│ + Linear(3584→3584)│  │ + Linear(3584→3584)│
└────────┬───────────┘  └────────┬───────────┘
         │                      │
         └──────────┬───────────┘
                    │ acoustic_embed + semantic_embed
                    ▼
         ┌──────────────────────────────────┐
         │  Insert at acoustic_input_mask    │
         │  positions in text embeddings     │
         │                                   │
         │  Text prompt format:              │
         │  <|im_start|>user                 │
         │  [context/hotwords]               │
         │  [AUDIO EMBEDDINGS HERE]          │
         │  <|im_end|>                       │
         │  <|im_start|>assistant            │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────────┐
         │  Qwen2 LLM (28 layers)           │
         │                                   │
         │  hidden_size: 3584                │
         │  num_attention_heads: 28          │
         │  num_key_value_heads: 4           │
         │  head_dim: 128                    │
         │  intermediate_size: 18944         │
         │  GQA repeat: 7                    │
         │  vocab_size: 152064               │
         │  rope_theta: 1,000,000            │
         │  rms_norm_eps: 1e-6               │
         │  max_position_embeddings: 32768   │
         │                                   │
         │  Autoregressive text generation   │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────────┐
         │  LM Head                          │
         │  Linear(3584 → 152064)            │
         │  → next token logits              │
         └──────────────────────────────────┘
                        │
                        ▼
         Output: JSON-structured transcription
         [{"Start": 0, "End": 15.43, "Speaker": 0, "Content": "..."}]
```

### Tokenizer Details

**Acoustic tokenizer** (σ-VAE):
- Encoder depths: [3, 3, 3, 3, 3, 3, 8] (7 stages)
- Downsampling ratios: [8, 5, 5, 4, 2, 2] → total 3200x
- 24kHz / 3200 = 7.5 Hz frame rate
- vae_dim=64, fix_std=0.5, gaussian sampling
- Causal convolutions for streaming support
- Long audio: processed in 60-second segments (1,440,000 samples)
  with `VibeVoiceTokenizerStreamingCache` to avoid conv overflow (>2^32)

**Semantic tokenizer** (encoder only):
- Same architecture, vae_dim=128, no VAE (fix_std=0, std_dist_type="none")
- Trained with ASR proxy task

### Audio Processing Pipeline

For audio longer than 60 seconds:
```
1. Split audio into 60-second segments
2. For each segment:
   a. Encode through acoustic tokenizer with streaming cache
   b. Encode through semantic tokenizer with streaming cache
   c. Append mean vectors
3. Concatenate all segment means
4. Sample from acoustic distribution (once, on concatenated means)
5. Project through connectors
6. Sum acoustic + semantic embeddings
```

## Performance

### Open ASR Leaderboard (WER %)

| Dataset | WER |
|---------|-----|
| librispeech_test.clean | 2.20 |
| tedlium_test | 2.57 |
| spgispeech_test | 3.80 |
| librispeech_test.other | 5.51 |
| voxpopuli_test | 8.01 |
| gigaspeech_test | 9.67 |
| earnings22_test | 13.17 |
| ami_test | 17.20 |
| **Average** | **7.77** |
| **RTFx** | **51.80** |

### Key Capabilities
- Speaker diarization (DER metric)
- Timestamp-level alignment (tcpWER metric)
- Hotword/context prompting for domain adaptation
- Code-switching across 50+ languages

## CoreML Export

### Scripts

| Script | Purpose |
|--------|---------|
| `convert_coreml.py` | Exports tokenizer encoders, connectors, LM head |
| `convert_stateful_lm.py` | Exports 28-layer Qwen2 with stateful KV cache (8B params) |

### Produced Artifacts

| File | Description | Size (est.) |
|------|-------------|-------------|
| `lm_decoder_stateful.mlpackage` | 28-layer Qwen2, 56 KV state buffers | ~14GB (fp16) |
| `acoustic_encoder.mlpackage` | σ-VAE encoder, variable-length audio | ~680MB |
| `semantic_encoder.mlpackage` | Semantic encoder, variable-length audio | ~680MB |
| `acoustic_connector.mlpackage` | Projects acoustic→LM space | ~50MB |
| `semantic_connector.mlpackage` | Projects semantic→LM space | ~50MB |
| `lm_head.mlpackage` | Hidden→logits (152064) | ~2.1GB |
| `embed_tokens.npy` | Token embeddings (152064, 3584) | ~2.1GB |

### Usage

```bash
cd models/asr/vibevoice-asr-hf/coreml
uv sync
uv run python convert_coreml.py --output-dir ./build/vibevoice-asr-hf
uv run python convert_stateful_lm.py --output-dir ./build/vibevoice-asr-hf
```

### Quantization Considerations

The 8B LLM backbone at fp16 is ~14GB — too large for most devices without
quantization. Recommended approach:

| Quantization | LLM Size | Quality Impact |
|-------------|----------|----------------|
| FP16 (baseline) | ~14GB | None |
| INT8 (linear) | ~7GB | Minimal |
| INT4 (palettized) | ~3.5GB | Moderate — test on ASR benchmarks |
| Mixed (INT4 LLM + FP16 encoders) | ~4.8GB | Best trade-off |

Use `coremltools.optimize.torch2coreml` for post-training quantization:
```python
import coremltools.optimize as cto
config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
)
compressed = cto.coreml.linear_quantize_weights(mlmodel, config)
```

### Runtime Notes

- **Audio encoders** accept variable-length input via `RangeDim` (1s to 60s)
- **Long audio**: Process in 60s segments on-device, concatenate features, then run LLM
- **Stateful LLM** requires iOS 18+ / macOS 15+
- **Prefill**: Feed audio embeddings + text prompt in one forward pass
- **Decode**: Single-token autoregressive generation until `<|im_end|>`
