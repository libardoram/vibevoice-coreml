# VibeVoice CoreML

On-device text-to-speech and speech-to-text using [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) models converted to CoreML.

Inspired by [FluidAudio](https://github.com/FluidInference/FluidAudio) with ideas from [Mobius](https://github.com/FluidInference/mobius).

## Models

| Model | Params | Type | HuggingFace |
|-------|--------|------|-------------|
| Realtime 0.5B | Qwen2.5-0.5B | Streaming TTS, 25 voices | [vibevoice-tts-0.5b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-0.5b-coreml) |
| Multispeaker 1.5B | Qwen2.5-1.5B | TTS + voice cloning | [vibevoice-tts-1.5b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-1.5b-coreml) |
| Multispeaker 7B | Qwen2.5-7B | TTS + voice cloning | [vibevoice-tts-7b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-7b-coreml) |
| ASR | Qwen2-7B (8.3B) | Speech-to-text, 50+ langs | [vibevoice-asr-coreml](https://huggingface.co/gafiatulin/vibevoice-asr-coreml) |

1.5B, 7B, and ASR models are INT8 quantized (W8A16) with fused LM+head. 0.5B runs at FP16. Models auto-download from HuggingFace on first use. All inference runs on CPU + GPU. Unfortunately I don't expect any step of any pipeline (even 0.5B) to run on ANE (better/faster than `CPU_GPU`).

## Performance

Benchmarked on M4 Max (64GB). RTF = real-time factor (higher is faster; >1x means faster than real-time). RAM measured per-process via `footprint`.

| Model | Backend | RTF | Peak RAM | Avg RAM | GPU | Power |
|-------|---------|-----|----------|---------|-----|-------|
| 0.5B | CoreML (Swift) | 5.7x | **3.5 GB** | 3.1 GB | **92%** | 40 W |
| 0.5B | MLX (Python) | **5.8x** | 5.9 GB | **2.2 GB** | 34% | **29 W** |
| 1.5B | CoreML (Swift) | 2.5x | **8.9 GB** | **7.7 GB** | **89%** | **49 W** |
| 1.5B | MLX (Python) | **3.7x** | 11 GB | 8.6 GB | 84% | 63 W |
| 7B | CoreML (Swift) | 1.2x | **17 GB** | **15 GB** | 82% | **69 W** |
| 7B | MLX (Python) | **1.4x** | 26 GB | 22 GB | **83%** | 88 W |
| ASR | CoreML (Swift) | 14.9 tok/s | 22 GB | 11 GB | 71% | **40 W** |
| ASR | MLX (Python) | **39.5 tok/s** | **17 GB** | **10 GB** | **88%** | 90 W |

### When to use what

- **Deploying in Swift apps (iOS/macOS)**: Use CoreML. It's the only option that integrates natively and runs without Python.
- **TTS 7B**: MLX is slightly faster (1.4x vs 1.2x RTF). CoreML uses 35% less RAM (17 vs 26 GB) and 22% less power — use CoreML for memory efficiency, MLX for speed.
- **TTS 1.5B**: MLX is 50% faster (3.7x vs 2.5x RTF). Memory is comparable (11 vs 9 GB peak). MLX is the better choice unless you need Swift integration.
- **TTS 0.5B**: Both are fast (5-6x RTF). CoreML is more memory-efficient.
- **ASR**: MLX is ~2.7x faster (40 vs 15 tok/s) but uses 2x the power. Use MLX for batch transcription, CoreML for on-device apps.
- **Memory-constrained (16 GB devices)**: 0.5B (3.5-6 GB) and 1.5B (9-11 GB) fit comfortably on either backend. 7B (17-26 GB) and ASR (17-22 GB) may work with swap pressure.

## Swift Library

Requires macOS 15+. Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/gafiatulin/vibevoice-coreml", from: "0.0.1"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "VibeVoiceCoreML", package: "vibevoice-coreml"),
        ]
    ),
]
```

### Text-to-Speech (0.5B Streaming)

```swift
import VibeVoiceCoreML

let tts = try await RealtimeTTS(modelsDir: modelsURL)
let config = TTSConfig(voice: "Emma")

for try await frame in tts.speak("Hello world", config: config) {
    // frame.samples: [Float] at 24kHz
}
```

### Text-to-Speech (1.5B / 7B with Voice Cloning)

```swift
import VibeVoiceCoreML

let tts = try await MultispeakerTTS(modelsDir: modelsURL, architecture: .model7B)
let voices = try await tts.encodeVoices(from: [referenceAudioURL])
let config = MultispeakerConfig(seed: 42)

for try await frame in tts.speak("Hello world", config: config, voices: voices) {
    // frame.samples: [Float] at 24kHz
}
```

### Speech-to-Text (ASR)

```swift
import VibeVoiceCoreML

let stt = try await SpeechToText(modelsDir: modelsURL)
let result = try await stt.transcribe(audioURL)
print(result.text)
```

### Auto-downloading Models

```swift
import VibeVoiceCoreML

let modelsURL = try await HubDownloader.download(repo: "gafiatulin/vibevoice-tts-0.5b-coreml")
```

## CLI

### Swift

```bash
swift build -c release
```

```bash
Usage: vibevoice-cli --model <0.5b|1.5b|7b|asr> [options]

Required:
  --model        Model variant: 0.5b, 1.5b, 7b, or asr

Optional:
  --models-dir   Directory with CoreML models (default: auto-download from HuggingFace)
  --cache-dir    HuggingFace hub cache directory (default: ~/.cache/huggingface/hub/)
  --verbose      Show detailed download and verification logs

TTS options:
  --tokenizer    Directory with tokenizer.json (default: models dir)
  --text         Text to synthesize (use \n for newlines)
  --output       Output WAV path (default: output.wav)
  --seed         Random seed (default: 42)

0.5B only:
  --voice        Voice name (default: Emma)

1.5B/7B only:
  --ref-audio    Reference audio file(s) for voice cloning (one per speaker)
  --max-tokens   Maximum speech tokens (default: auto from text length)

ASR options:
  --audio        Input audio file to transcribe
  --prompt       Optional prompt/context to guide transcription
  --max-tokens   Maximum tokens to generate (default: auto from audio duration)

Examples:
  vibevoice-cli --model 0.5b --text "Hello world"
  vibevoice-cli --model 7b --ref-audio spk1.wav --text "Hello from a cloned voice"
  vibevoice-cli --model asr --audio recording.wav
  vibevoice-cli --model 0.5b --models-dir /path/to/models --text "Use local models"
```

### Python

```bash
cd python/python-cli
uv sync
```

```bash
usage: vibevoice_cli.py [-h] [--model {0.5b,1.5b,7b,asr}] [--models-dir MODELS_DIR] [--cache-dir CACHE_DIR] [--verbose] [--text TEXT] [--output OUTPUT] [--seed SEED] [--voice VOICE]
                        [--ref-audio REF_AUDIO [REF_AUDIO ...]] [--max-tokens MAX_TOKENS] [--audio AUDIO] [--prompt PROMPT]

VibeVoice CoreML CLI — TTS and ASR inference

options:
  -h, --help            show this help message and exit
  --model {0.5b,1.5b,7b,asr}
                        Model variant: 0.5b, 1.5b, 7b, or asr
  --models-dir MODELS_DIR
                        Local models directory (default: auto-download from HuggingFace)
  --cache-dir CACHE_DIR
                        Cache directory for downloaded models
  --verbose, -v         Show detailed logs
  --text TEXT           Text to synthesize
  --output, -o OUTPUT   Output file path
  --seed SEED           Random seed (default: 42)
  --voice VOICE         Voice name (0.5B only, default: Emma)
  --ref-audio REF_AUDIO [REF_AUDIO ...]
                        Reference audio file(s) for voice cloning (1.5B/7B only)
  --max-tokens MAX_TOKENS
                        Maximum speech/generation tokens (default: auto)
  --audio AUDIO         Input audio file for ASR
  --prompt PROMPT       Optional ASR context prompt

Examples:
  vibevoice-cli --model 0.5b --text "Hello world"
  vibevoice-cli --model 7b --ref-audio spk1.wav --text "Hello from a cloned voice"
  vibevoice-cli --model asr --audio recording.wav
  vibevoice-cli --model 0.5b --models-dir /path/to/models --text "Use local models"
```

Both CLIs auto-download models from HuggingFace and share the same cache (`~/.cache/huggingface/hub/`).

## Project Structure

```
Package.swift                     # Swift package
Sources/
  VibeVoiceCoreML/                # Swift library (RealtimeTTS, MultispeakerTTS, SpeechToText)
  vibevoice-cli/                  # Swift CLI
python/
  common/                         # Shared conversion utilities
  tts/                            # TTS model conversion & pipelines
    common/                       # TTS-specific utilities (RoPE, diffusion, etc.)
    vibevoice-realtime-0.5b/      # 0.5B streaming model
    vibevoice-multispeaker/       # 1.5B + 7B multispeaker models
  stt/                            # ASR model conversion & pipelines
  python-cli/                     # Python CLI
  build_hf_repos.py               # HuggingFace repo builder
```

## Converting Models

To convert models from PyTorch to CoreML yourself:

```bash
cd python/tts/vibevoice-multispeaker
uv sync
uv run python convert/convert_all.py --model-id 7b --int8 --fuse-lm-head
```

## License

MIT
