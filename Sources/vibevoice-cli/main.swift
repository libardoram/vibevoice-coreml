import Foundation
import VibeVoiceCoreML

// MARK: - WAV writer

func writeWAV(samples: [Float], sampleRate: Int = 24000, to url: URL) throws {
    let numSamples = samples.count
    let bytesPerSample = 2 // int16
    let dataSize = numSamples * bytesPerSample
    let fileSize = 44 + dataSize

    var data = Data(capacity: fileSize)

    // RIFF header
    data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
    data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize - 8).littleEndian) { Array($0) })
    data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"

    // fmt chunk
    data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * bytesPerSample).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(bytesPerSample).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })  // bits

    // data chunk
    data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
    data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        let int16Val = Int16(clamped * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: int16Val.littleEndian) { Array($0) })
    }

    try data.write(to: url)
}

// MARK: - Model configs

struct ModelConfig {
    let name: String
    let architecture: ModelArchitecture?  // nil = 0.5B realtime or ASR
    let hfRepo: String

    static let all: [String: ModelConfig] = [
        "0.5b": ModelConfig(
            name: "0.5B Realtime",
            architecture: nil,
            hfRepo: "gafiatulin/vibevoice-tts-0.5b-coreml"
        ),
        "1.5b": ModelConfig(
            name: "1.5B Multispeaker",
            architecture: .model1_5B,
            hfRepo: "gafiatulin/vibevoice-tts-1.5b-coreml"
        ),
        "7b": ModelConfig(
            name: "7B Multispeaker",
            architecture: .model7B,
            hfRepo: "gafiatulin/vibevoice-tts-7b-coreml"
        ),
        "asr": ModelConfig(
            name: "ASR 7B",
            architecture: nil,
            hfRepo: "gafiatulin/vibevoice-asr-coreml"
        ),
    ]
}

// MARK: - CLI

func printUsage() {
    print("""
    Usage: vibevoice-cli --model <0.5b|1.5b|7b|asr> [options]

    Required:
      --model        Model variant: 0.5b, 1.5b, 7b, or asr

    Optional:
      --models-dir   Directory with CoreML models (default: auto-download from HuggingFace)
      --cache-dir    HuggingFace hub cache directory (default: ~/.cache/huggingface/hub/)
      --verbose      Show detailed download and verification logs

    TTS options:
      --tokenizer    Directory with tokenizer.json (default: models dir)
      --text         Text to synthesize (use \\n for newlines)
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
    """)
}

// MARK: - Entry point

let args = CommandLine.arguments

// Parse all arguments
var modelName = ""
var modelsPath = ""
var cachePath = ""
var voicesPath = ""
var tokenizerPath = ""
var text = ""
var voice = "Emma"
var outputPath = ""
var seed: UInt64? = 42
var maxTokens: Int? = nil
var refAudioPaths: [URL] = []
var audioPath = ""
var prompt: String? = nil
var verbose = false

var i = 1
while i < args.count {
    switch args[i] {
    case "--model": i += 1; modelName = args[i].lowercased()
    case "--models-dir": i += 1; modelsPath = args[i]
    case "--models": i += 1; modelsPath = args[i]  // backward compat
    case "--cache-dir": i += 1; cachePath = args[i]
    case "--voices": i += 1; voicesPath = args[i]
    case "--tokenizer": i += 1; tokenizerPath = args[i]
    case "--text": i += 1; text = args[i].replacingOccurrences(of: "\\n", with: "\n")
    case "--voice": i += 1; voice = args[i]
    case "--output": i += 1; outputPath = args[i]
    case "--seed": i += 1; seed = UInt64(args[i])
    case "--max-tokens": i += 1; maxTokens = Int(args[i])
    case "--audio": i += 1; audioPath = args[i]
    case "--prompt": i += 1; prompt = args[i]
    case "--verbose", "-v": verbose = true
    case "--ref-audio":
        i += 1
        while i < args.count && !args[i].hasPrefix("--") {
            refAudioPaths.append(URL(fileURLWithPath: args[i]))
            i += 1
        }
        continue
    // Backward compat: old-style flags
    case "--multispeaker":
        if modelName.isEmpty { modelName = "1.5b" }
    case "--7b":
        modelName = "7b"
    case "--help", "-h":
        printUsage()
        Foundation.exit(0)
    default: break
    }
    i += 1
}

if modelName.isEmpty {
    printUsage()
    Foundation.exit(0)
}

guard let config = ModelConfig.all[modelName] else {
    print("Error: unknown model '\(modelName)'. Choose from: \(ModelConfig.all.keys.sorted().joined(separator: ", "))")
    Foundation.exit(1)
}

// Validate required args before downloading
if modelName == "asr" && audioPath.isEmpty {
    print("Error: --audio is required for ASR mode")
    Foundation.exit(1)
}
if modelName != "asr" && text.isEmpty {
    print("Error: --text is required for TTS mode")
    Foundation.exit(1)
}

// Resolve models directory: explicit path or auto-download from HuggingFace
let modelsURL: URL
if !modelsPath.isEmpty {
    modelsURL = URL(fileURLWithPath: modelsPath)
} else {
    let cacheDir = cachePath.isEmpty ? nil : URL(fileURLWithPath: cachePath)
    print("Syncing \(config.name) from \(config.hfRepo)...")
    fflush(stdout)
    let verboseLog: (@Sendable (_ message: String) -> Void)?
    if verbose {
        verboseLog = { msg in print("  [verbose] \(msg)"); fflush(stdout) }
    } else {
        verboseLog = nil
    }
    do {
        modelsURL = try await HubDownloader.download(
            repo: config.hfRepo,
            to: cacheDir,
            log: verboseLog
        )
        print("Models ready: \(modelsURL.path)")
    } catch {
        print("Error downloading models: \(error)")
        Foundation.exit(1)
    }
}

// Resolve tokenizer: models dir (bundled), explicit --tokenizer, or error
if tokenizerPath.isEmpty {
    let bundled = modelsURL.appendingPathComponent("tokenizer.json")
    if FileManager.default.fileExists(atPath: bundled.path) {
        tokenizerPath = modelsURL.path
    }
}
if tokenizerPath.isEmpty {
    print("Error: tokenizer not found. Place tokenizer.json in models dir or use --tokenizer <dir>")
    Foundation.exit(1)
}


let tokenizerURL = URL(fileURLWithPath: tokenizerPath)

do {
    if modelName == "asr" {
        // ASR mode
        if audioPath.isEmpty {
            print("Error: --audio is required for ASR mode")
            Foundation.exit(1)
        }
        let audioURL = URL(fileURLWithPath: audioPath)

        print("Loading ASR models...")
        fflush(stdout)
        let t0 = CFAbsoluteTimeGetCurrent()
        let stt = try SpeechToText(modelsDir: modelsURL, tokenizerDir: tokenizerURL)
        print("  Loaded in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")

        let sttConfig = STTConfig(maxNewTokens: maxTokens, prompt: prompt)
        print("Transcribing: \(audioURL.lastPathComponent)")
        fflush(stdout)

        let result = try await stt.transcribe(audioURL, config: sttConfig)
        print("Audio duration: \(String(format: "%.1f", result.audioDuration))s")
        print("Generated \(result.tokensGenerated) tokens in \(String(format: "%.1f", result.generationTime))s (\(String(format: "%.1f", Double(result.tokensGenerated) / result.generationTime)) tok/s)")

        // Write to file
        let outURL: URL
        if !outputPath.isEmpty {
            outURL = URL(fileURLWithPath: outputPath)
        } else {
            let audioName = audioURL.deletingPathExtension().lastPathComponent
            let ts = DateFormatter()
            ts.dateFormat = "yyyyMMdd_HHmmss"
            outURL = URL(fileURLWithPath: "\(ts.string(from: Date()))_\(audioName)_swift.txt")
        }
        try result.text.write(to: outURL, atomically: true, encoding: .utf8)
        print("Saved to \(outURL.path)")

    } else if let arch = config.architecture {
        // 1.5B / 7B multispeaker
        print("Loading \(config.name) models...")
        fflush(stdout)
        let t0 = CFAbsoluteTimeGetCurrent()
        let tts = try MultispeakerTTS(
            modelsDir: modelsURL, tokenizerDir: tokenizerURL,
            architecture: arch
        )
        print("  Loaded in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")

        var voices: [VoiceReference]? = nil
        if !refAudioPaths.isEmpty {
            for (i, path) in refAudioPaths.enumerated() {
                print("Voice cloning Speaker \(i + 1): \(path.lastPathComponent)")
            }
            let t_enc = CFAbsoluteTimeGetCurrent()
            voices = try tts.encodeVoices(from: refAudioPaths)
            print("  Encoded \(refAudioPaths.count) voice(s) in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t_enc))s")
        }

        let msConfig = MultispeakerConfig(maxSpeechTokens: maxTokens, seed: seed)
        print("Generating: \"\(text)\"")
        fflush(stdout)

        let t1 = CFAbsoluteTimeGetCurrent()
        var allSamples: [Float] = []
        var frameCount = 0
        for try await frame in tts.speak(text, config: msConfig, voices: voices) {
            allSamples.append(contentsOf: frame.samples)
            frameCount += 1
        }

        let genTime = CFAbsoluteTimeGetCurrent() - t1
        let audioDuration = Double(allSamples.count) / 24000.0
        let rtf = audioDuration / genTime
        print("Generated \(frameCount) frames, \(String(format: "%.1f", audioDuration))s audio in \(String(format: "%.1f", genTime))s (\(String(format: "%.2f", rtf))x RTF)")

        let wavPath = outputPath.isEmpty ? "output.wav" : outputPath
        try writeWAV(samples: allSamples, to: URL(fileURLWithPath: wavPath))
        print("Saved to \(wavPath)")

    } else {
        // 0.5B realtime
        let defaultVoices = modelsURL.appendingPathComponent("voices")
        let voicesURL = voicesPath.isEmpty
            ? (FileManager.default.fileExists(atPath: defaultVoices.path) ? defaultVoices : modelsURL)
            : URL(fileURLWithPath: voicesPath)

        print("Loading \(config.name) models...")
        fflush(stdout)
        let t0 = CFAbsoluteTimeGetCurrent()
        let tts = try RealtimeTTS(modelsDir: modelsURL, voicesDir: voicesURL, tokenizerDir: tokenizerURL)
        print("  Loaded in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")

        let rtConfig = TTSConfig(voice: voice, seed: seed)
        print("Generating: \"\(text)\"")
        fflush(stdout)

        let t1 = CFAbsoluteTimeGetCurrent()
        var allSamples: [Float] = []
        var frameCount = 0
        for try await frame in tts.speak(text, config: rtConfig) {
            allSamples.append(contentsOf: frame.samples)
            frameCount += 1
        }

        let genTime = CFAbsoluteTimeGetCurrent() - t1
        let audioDuration = Double(allSamples.count) / 24000.0
        let rtf = audioDuration / genTime
        print("Generated \(frameCount) frames, \(String(format: "%.1f", audioDuration))s audio in \(String(format: "%.1f", genTime))s (\(String(format: "%.2f", rtf))x RTF)")

        let wavPath = outputPath.isEmpty ? "output.wav" : outputPath
        try writeWAV(samples: allSamples, to: URL(fileURLWithPath: wavPath))
        print("Saved to \(wavPath)")
    }
} catch {
    print("ERROR: \(error)")
    Foundation.exit(1)
}
