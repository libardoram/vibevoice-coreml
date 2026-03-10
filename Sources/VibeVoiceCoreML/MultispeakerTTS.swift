import AVFoundation
import CoreML
import Foundation
import Tokenizers

// MARK: - Multispeaker TTS Configuration

/// Configuration for multispeaker TTS generation (1.5B/7B models).
public struct MultispeakerConfig: Sendable {
    /// Classifier-free guidance scale.
    public var cfgScale: Float
    /// Maximum speech tokens. nil = auto-estimate from text (~4.5 tokens/word).
    public var maxSpeechTokens: Int?
    /// Random seed for reproducible output. nil = random.
    public var seed: UInt64?

    public init(
        cfgScale: Float = 1.3,
        maxSpeechTokens: Int? = nil,
        seed: UInt64? = nil
    ) {
        self.cfgScale = cfgScale
        self.maxSpeechTokens = maxSpeechTokens
        self.seed = seed
    }
}

/// Pre-encoded voice reference for voice cloning. Encode once, reuse across generations.
public struct VoiceReference: Sendable {
    let speakerID: Int
    let numVAETokens: Int
    /// Pre-computed embeddings (numVAETokens * hiddenSize floats).
    let embeddings: [Float]
}

/// Architecture parameters for multispeaker models.
public struct ModelArchitecture: Sendable {
    public let hiddenSize: Int
    public let headDim: Int
    public let numQHeads: Int
    public let numKVHeads: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let semanticDim: Int

    /// 1.5B model (Qwen2.5-1.5B backbone).
    public static let model1_5B = ModelArchitecture(
        hiddenSize: 1536, headDim: 128, numQHeads: 12,
        numKVHeads: 2, numLayers: 28, vocabSize: 151936, semanticDim: 128
    )

    /// 7B model (Qwen2.5-7B backbone).
    public static let model7B = ModelArchitecture(
        hiddenSize: 3584, headDim: 128, numQHeads: 28,
        numKVHeads: 4, numLayers: 28, vocabSize: 152064, semanticDim: 128
    )
}

// MARK: - Multispeaker TTS Engine

/// Multispeaker TTS using 1.5B or 7B models.
public final class MultispeakerTTS: @unchecked Sendable {
    private let arch: ModelArchitecture
    private let lm: MLModel
    private let diffusionLoop: MLModel
    private let vaeDecoder: MLModel
    private let acousticConnector: MLModel
    private let semanticConnector: MLModel
    private let semanticEncoder: MLModel
    private let vaeEncoder: MLModel?
    private let embedTokens: EmbeddingTable
    private let tokenizerDir: URL

    // Special token IDs
    static let speechStartID = 151652
    static let speechEndID = 151653
    static let speechDiffusionID = 151654
    static let eosID = 151643

    // Scaling factors (1.5B/7B use inverse direction vs 0.5B)
    static let speechScaling: Float = 0.1962890625
    static let speechBias: Float = -0.04931640625

    public init(
        modelsDir: URL,
        tokenizerDir: URL,
        architecture: ModelArchitecture = .model1_5B
    ) throws {
        self.arch = architecture
        self.tokenizerDir = tokenizerDir

        let configCPUGPU = MLModelConfiguration()
        configCPUGPU.computeUnits = .cpuAndGPU
        let configAll = MLModelConfiguration()

        func load(_ name: String, config cfg: MLModelConfiguration = configAll) throws -> MLModel {
            let compiledURL = modelsDir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL, configuration: cfg)
            }
            let packageURL = modelsDir.appendingPathComponent("\(name).mlpackage")
            let compiledDir = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledDir, configuration: cfg)
        }

        lm = try load("lm_decoder_fused_int8", config: configCPUGPU)
        diffusionLoop = try load("diffusion_loop")
        vaeDecoder = try load("vae_decoder_streaming", config: configCPUGPU)
        acousticConnector = try load("acoustic_connector")
        semanticConnector = try load("semantic_connector")
        semanticEncoder = try load("semantic_encoder_streaming", config: configCPUGPU)

        let vaeEncURL = modelsDir.appendingPathComponent("vae_encoder.mlmodelc")
        let vaeEncPkgURL = modelsDir.appendingPathComponent("vae_encoder.mlpackage")
        if FileManager.default.fileExists(atPath: vaeEncURL.path) {
            vaeEncoder = try MLModel(contentsOf: vaeEncURL, configuration: configAll)
        } else if FileManager.default.fileExists(atPath: vaeEncPkgURL.path) {
            let compiled = try MLModel.compileModel(at: vaeEncPkgURL)
            vaeEncoder = try MLModel(contentsOf: compiled, configuration: configAll)
        } else {
            vaeEncoder = nil
        }

        embedTokens = try EmbeddingTable(
            contentsOf: modelsDir.appendingPathComponent("embed_tokens.bin")
        )
    }

    /// Encode reference audio files into reusable voice references.
    /// Speaker 1 in text maps to the first URL, Speaker 2 to the second, etc.
    public func encodeVoices(from urls: [URL]) throws -> [VoiceReference] {
        return try urls.enumerated().map { (i, url) in
            try encodeSingleVoice(url: url, speakerID: i)
        }
    }

    /// Generate speech from text, streaming audio frames.
    public func speak(
        _ text: String,
        config: MultispeakerConfig = MultispeakerConfig(),
        voices: [VoiceReference]? = nil
    ) -> AsyncThrowingStream<AudioFrame, Error> {
        let generator = self
        return AsyncThrowingStream { continuation in
            Task { @Sendable in
                do {
                    try await generator.generate(
                        text: text, config: config, voices: voices, continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Generate speech and return all audio at once.
    public func synthesize(
        _ text: String,
        config: MultispeakerConfig = MultispeakerConfig(),
        voices: [VoiceReference]? = nil
    ) async throws -> [Float] {
        var allSamples: [Float] = []
        for try await frame in speak(text, config: config, voices: voices) {
            allSamples.append(contentsOf: frame.samples)
        }
        return allSamples
    }
}

// MARK: - Synchronous CoreML helpers (avoid async overload selection)

private func predictLM(
    model: MLModel, state: MLState,
    hidden: [Float], positions: [Int],
    hiddenSize: Int, headDim: Int, seqLen: Int, totalLen: Int
) throws -> (hidden: [Float], logits: [Float]) {
    let rope = computeRoPE(positions: positions, headDim: headDim)

    var mask = [Float](repeating: 0, count: seqLen * totalLen)
    if seqLen == totalLen {
        for i in 0 ..< seqLen {
            for j in (i + 1) ..< seqLen {
                mask[i * totalLen + j] = -1e9
            }
        }
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_states": makeMultiArray(hidden, shape: [1, seqLen, hiddenSize]),
        "position_cos": makeMultiArray(rope.cos, shape: [1, seqLen, headDim]),
        "position_sin": makeMultiArray(rope.sin, shape: [1, seqLen, headDim]),
        "attention_mask": makeMultiArray(mask, shape: [1, 1, seqLen, totalLen]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input, using: state)
    let h = readMultiArray(output.featureValue(for: "output_hidden")!.multiArrayValue!)
    let l = readMultiArray(output.featureValue(for: "logits")!.multiArrayValue!)
    return (h, l)
}

/// Decode-only LM step (q=1): zero-copy argmax on logits.
private func decodeLMStep(
    model: MLModel, state: MLState,
    hidden: [Float], position: Int,
    hiddenSize: Int, headDim: Int, vocabSize: Int
) throws -> (hidden: [Float], token: Int) {
    let totalLen = position + 1
    let rope = computeRoPE(positions: [position], headDim: headDim)

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_states": makeMultiArray(hidden, shape: [1, 1, hiddenSize]),
        "position_cos": makeMultiArray(rope.cos, shape: [1, 1, headDim]),
        "position_sin": makeMultiArray(rope.sin, shape: [1, 1, headDim]),
        "attention_mask": makeMultiArray([Float](repeating: 0, count: totalLen), shape: [1, 1, 1, totalLen]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input, using: state)
    let h = readMultiArray(output.featureValue(for: "output_hidden")!.multiArrayValue!)
    let logitsArr = output.featureValue(for: "logits")!.multiArrayValue!
    let token = argmaxMultiArray(logitsArr, lastTokenOffset: 0, count: vocabSize)
    return (h, token)
}

private func predictDiffusion(
    model: MLModel, noise: [Float], condition: [Float],
    negCondition: [Float], cfgScale: Float, hiddenSize: Int
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "noise": makeMultiArray(noise, shape: [1, Constants.vaeDim]),
        "condition": makeMultiArray(condition, shape: [1, hiddenSize]),
        "neg_condition": makeMultiArray(negCondition, shape: [1, hiddenSize]),
        "cfg_scale": makeMultiArray([cfgScale], shape: [1]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input)
    return readMultiArray(output.featureValue(for: "latent")!.multiArrayValue!)
}

private func predictVAE(
    model: MLModel, state: MLState, latent: [Float]
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "latent": makeMultiArray(latent, shape: [1, Constants.vaeDim, 1]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input, using: state)
    return readMultiArray(output.featureValue(for: "audio")!.multiArrayValue!)
}

private func predictConnector(
    model: MLModel, latent: [Float], dim: Int
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "speech_latent": makeMultiArray(latent, shape: [1, 1, dim]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input)
    return readMultiArray(output.featureValue(for: "embedding")!.multiArrayValue!)
}

private func predictSemanticEncoder(
    model: MLModel, state: MLState, audio: [Float]
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "audio": makeMultiArray(audio, shape: [1, 1, 3200]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input, using: state)
    return readMultiArray(output.featureValue(for: "features")!.multiArrayValue!)
}

private func predictSemanticConnector(
    model: MLModel, features: [Float], dim: Int
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "semantic_features": makeMultiArray(features, shape: [1, 1, dim]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input)
    return readMultiArray(output.featureValue(for: "embedding")!.multiArrayValue!)
}

// MARK: - Voice cloning helpers

/// Internal: per-speaker data with embed positions for prompt injection.
private struct SpeakerPromptData {
    let ref: VoiceReference
    let embedPositions: [Int]
}

/// Result of tokenizing a prompt with voice cloning.
private struct VoiceClonePrompt {
    let inputIds: [Int]
    let speakers: [SpeakerPromptData]
}

private let voiceCloneSamples = 240000  // 10s at 24kHz
private let speechTokCompressRatio = 3200

private func predictVAEEncoder(
    model: MLModel, audio: [Float]
) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "audio": makeMultiArray(audio, shape: [1, 1, voiceCloneSamples]),
    ] as [String: MLMultiArray])
    let output = try model.prediction(from: input)
    let arr = output.featureValue(for: "latent")!.multiArrayValue!
    // VAE encoder output may have non-contiguous strides — use subscript access
    let dims = arr.shape.map { $0.intValue }  // [1, vaeDim, T]
    let vaeDim = dims[1]
    let tDim = dims[2]
    var result = [Float](repeating: 0, count: vaeDim * tDim)
    for d in 0 ..< vaeDim {
        for t in 0 ..< tDim {
            result[d * tDim + t] = arr[[0, d, t] as [NSNumber]].floatValue
        }
    }
    return result
}

/// Load audio file (WAV, MP3, M4A, .raw) as mono float32 samples at 24kHz.
/// .raw files are read directly as little-endian float32 (for exact cross-platform matching).
private func loadAudio(url: URL) throws -> [Float] {
    if url.pathExtension == "raw" {
        let data = try Data(contentsOf: url)
        let count = data.count / 4
        return data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self).prefix(count))
        }
    }
    let file = try AVAudioFile(forReading: url)
    let srcFormat = file.processingFormat
    let frameCount = AVAudioFrameCount(file.length)

    guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: frameCount) else {
        throw VibeVoiceError.invalidData("Failed to create audio buffer")
    }
    try file.read(into: srcBuffer)

    // Convert to mono 24kHz float32
    let targetRate: Double = 24000
    guard let monoFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: targetRate,
        channels: 1,
        interleaved: false
    ) else {
        throw VibeVoiceError.invalidData("Failed to create target audio format")
    }

    guard let converter = AVAudioConverter(from: srcFormat, to: monoFormat) else {
        throw VibeVoiceError.invalidData("Failed to create audio converter")
    }

    let ratio = targetRate / srcFormat.sampleRate
    let outputFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: outputFrames) else {
        throw VibeVoiceError.invalidData("Failed to create output buffer")
    }

    var error: NSError?
    converter.convert(to: outputBuffer, error: &error) { _, outStatus in
        outStatus.pointee = .haveData
        return srcBuffer
    }
    if let error { throw error }

    guard let channelData = outputBuffer.floatChannelData else {
        throw VibeVoiceError.invalidData("No audio data after conversion")
    }
    let count = Int(outputBuffer.frameLength)
    return Array(UnsafeBufferPointer(start: channelData[0], count: count))
}

// MARK: - Internal generation

extension MultispeakerTTS {
    /// Encode a single reference audio file into a VoiceReference.
    private func encodeSingleVoice(url: URL, speakerID: Int) throws -> VoiceReference {
        guard let encoder = vaeEncoder else {
            throw VibeVoiceError.modelError("VAE encoder not found (required for voice cloning)")
        }
        let hs = arch.hiddenSize

        var wav = try loadAudio(url: url)
        if wav.count > voiceCloneSamples {
            wav = Array(wav[0 ..< voiceCloneSamples])
        }
        let numVAETokens = (wav.count + speechTokCompressRatio - 1) / speechTokCompressRatio

        // VAE encode: pad/trim to 240000 samples
        var audioInput = [Float](repeating: 0, count: voiceCloneSamples)
        let copyLen = min(wav.count, voiceCloneSamples)
        for i in 0 ..< copyLen { audioInput[i] = wav[i] }

        let latents = try predictVAEEncoder(model: encoder, audio: audioInput)
        let tFull = latents.count / Constants.vaeDim
        let actualT = min(tFull, numVAETokens)

        // Apply scaling/bias and connect frame-by-frame
        var allEmbeds = [Float](repeating: 0, count: actualT * hs)
        for t in 0 ..< actualT {
            var frame = [Float](repeating: 0, count: Constants.vaeDim)
            for d in 0 ..< Constants.vaeDim {
                let raw = latents[d * tFull + t]
                frame[d] = (raw + Self.speechBias) * Self.speechScaling
            }
            let emb = try predictConnector(
                model: acousticConnector, latent: frame, dim: Constants.vaeDim
            )
            for j in 0 ..< hs {
                allEmbeds[t * hs + j] = emb[j]
            }
        }

        return VoiceReference(speakerID: speakerID, numVAETokens: numVAETokens, embeddings: allEmbeds)
    }

    /// Build voice cloning prompt from pre-encoded voices.
    private func buildVoiceClonePrompt(
        voices: [VoiceReference], tokenizer: any Tokenizer
    ) -> VoiceClonePrompt {
        let systemPrompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        let systemTokens = tokenizer.encode(text: systemPrompt)
        let voicePrefix = tokenizer.encode(text: " Voice input:\n")
        let newlineTok = tokenizer.encode(text: "\n")

        var voiceTokens = voicePrefix
        var speakers: [SpeakerPromptData] = []
        var currentOffset = systemTokens.count + voicePrefix.count

        for voice in voices {
            let spkPrefix = tokenizer.encode(text: " Speaker \(voice.speakerID):")
            voiceTokens += spkPrefix
            currentOffset += spkPrefix.count

            voiceTokens.append(Self.speechStartID)
            currentOffset += 1

            let embedPositions = Array(currentOffset ..< currentOffset + voice.numVAETokens)
            voiceTokens += [Int](repeating: Self.speechDiffusionID, count: voice.numVAETokens)
            currentOffset += voice.numVAETokens

            voiceTokens.append(Self.speechEndID)
            voiceTokens += newlineTok
            currentOffset += 1 + newlineTok.count

            speakers.append(SpeakerPromptData(ref: voice, embedPositions: embedPositions))
        }

        return VoiceClonePrompt(inputIds: systemTokens + voiceTokens, speakers: speakers)
    }

    private func generate(
        text: String,
        config: MultispeakerConfig,
        voices: [VoiceReference]?,
        continuation: AsyncThrowingStream<AudioFrame, Error>.Continuation
    ) async throws {
        let hs = arch.hiddenSize
        let hd = arch.headDim

        // Tokenize
        let tokenizer = try await TokenizerLoader.load(from: tokenizerDir)

        let systemPrompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

        var inputIds: [Int]
        var voiceClone: VoiceClonePrompt? = nil

        if let voices, !voices.isEmpty {
            // Voice cloning path
            let vc = buildVoiceClonePrompt(voices: voices, tokenizer: tokenizer)
            voiceClone = vc

            // Text section with speaker ID normalization (1-based → 0-based)
            let textSection = tokenizer.encode(text: " Text input:\n")
            var speakerTokens: [Int] = []
            for line in text.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n") {
                var lineStr = String(line)
                if let range = lineStr.range(of: #"Speaker\s+(\d+)"#, options: .regularExpression) {
                    let match = lineStr[range]
                    if let numRange = match.range(of: #"\d+"#, options: .regularExpression) {
                        let n = Int(match[numRange]) ?? 1
                        lineStr.replaceSubrange(range, with: "Speaker \(max(0, n - 1))")
                    }
                }
                speakerTokens += tokenizer.encode(text: " \(lineStr)\n")
            }
            let outputSection = tokenizer.encode(text: " Speech output:\n")
            inputIds = vc.inputIds + textSection + speakerTokens + outputSection
                + [Self.speechStartID]
        } else {
            // Standard path (no voice cloning)
            let systemTokens = tokenizer.encode(text: systemPrompt)
            let textSection = tokenizer.encode(text: " Text input:\n")
            var speakerTokens: [Int] = []
            for line in text.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n") {
                speakerTokens += tokenizer.encode(text: " \(line)\n")
            }
            let outputSection = tokenizer.encode(text: " Speech output:\n")
            inputIds = systemTokens + textSection + speakerTokens + outputSection
                + [Self.speechStartID]
        }

        // Estimate max speech tokens: strip "Speaker N:" prefixes, ~4.5 tokens/word
        let textOnly = text.replacingOccurrences(
            of: #"Speaker\s+\d+\s*:"#, with: "", options: .regularExpression
        )
        let wordCount = textOnly.split(whereSeparator: { $0.isWhitespace }).count
        let tokenLimit = config.maxSpeechTokens ?? max(Int(Double(wordCount) * 4.5), 20)

        // Build prefill embeddings (inject voice cloning embeddings at placeholder positions)
        let state = lm.makeState()
        let seqLen = inputIds.count

        // Compute negative condition: speech_start through a fresh LM state
        let negState = lm.makeState()
        let (negH, _) = try decodeLMStep(
            model: lm, state: negState,
            hidden: embedTokens.lookup([Self.speechStartID]),
            position: 0, hiddenSize: hs, headDim: hd, vocabSize: arch.vocabSize
        )
        let negCond = Array(negH[0 ..< hs])
        var prefillEmbeds = embedTokens.lookup(inputIds)  // (seqLen * hs)

        if let vc = voiceClone {
            for spk in vc.speakers {
                for (i, pos) in spk.embedPositions.enumerated() {
                    if i < spk.ref.numVAETokens {
                        let srcOff = i * hs
                        let dstOff = pos * hs
                        for j in 0 ..< hs {
                            prefillEmbeds[dstOff + j] = spk.ref.embeddings[srcOff + j]
                        }
                    }
                }
            }
        }

        // Prefill (still uses full predictLM — only once, so logits copy is fine)
        let (prefillH, prefillL) = try predictLM(
            model: lm, state: state,
            hidden: prefillEmbeds,
            positions: Array(0 ..< seqLen),
            hiddenSize: hs, headDim: hd, seqLen: seqLen, totalLen: seqLen
        )

        let lastOff = (seqLen - 1) * hs
        var currentHidden = Array(prefillH[lastOff ..< lastOff + hs])
        var nextToken = argmax(prefillL, offset: (seqLen - 1) * arch.vocabSize, count: arch.vocabSize)

        var vaeState = vaeDecoder.makeState()
        var semState = semanticEncoder.makeState()
        var rng = NumpyRNG(seed: config.seed ?? UInt64.random(in: 0 ..< UInt64.max))

        var position = seqLen
        var speechTokenCount = 0
        var frameIndex = 0
        var lastAudioChunk: [Float]? = nil

        for _ in 0 ..< tokenLimit * 3 {
            if nextToken == Self.eosID { break }
            if speechTokenCount >= tokenLimit { break }

            var nextEmbed: [Float]

            if nextToken == Self.speechDiffusionID {
                speechTokenCount += 1

                // Diffusion — match Python's randint(0, 2**31) for inner seed
                let innerSeed = UInt64(rng.randint(UInt32(1) << 31))
                var innerRng = NumpyRNG(seed: innerSeed)
                let noise = innerRng.randn(Constants.vaeDim)
                let speechLatent = try predictDiffusion(
                    model: diffusionLoop, noise: noise,
                    condition: currentHidden, negCondition: negCond,
                    cfgScale: config.cfgScale, hiddenSize: hs
                )
                // Scale and decode via streaming VAE
                var scaledLatent = [Float](repeating: 0, count: Constants.vaeDim)
                for i in 0 ..< Constants.vaeDim {
                    scaledLatent[i] = speechLatent[i] / Self.speechScaling - Self.speechBias
                }
                let audioFrame = try predictVAE(
                    model: vaeDecoder, state: vaeState, latent: scaledLatent
                )
                continuation.yield(AudioFrame(samples: audioFrame, index: frameIndex))
                frameIndex += 1
                lastAudioChunk = audioFrame

                // Acoustic + semantic connector
                let acousticEmbed = try predictConnector(
                    model: acousticConnector, latent: speechLatent, dim: Constants.vaeDim
                )

                if let chunk = lastAudioChunk {
                    var audioInput = [Float](repeating: 0, count: 3200)
                    let copyLen = min(chunk.count, 3200)
                    for i in 0 ..< copyLen { audioInput[i] = chunk[i] }

                    let features = try predictSemanticEncoder(
                        model: semanticEncoder, state: semState, audio: audioInput
                    )
                    let semanticEmbed = try predictSemanticConnector(
                        model: semanticConnector, features: features, dim: arch.semanticDim
                    )

                    nextEmbed = [Float](repeating: 0, count: hs)
                    for j in 0 ..< hs {
                        nextEmbed[j] = acousticEmbed[j] + semanticEmbed[j]
                    }
                } else {
                    nextEmbed = Array(acousticEmbed[0 ..< hs])
                }
            } else {
                if nextToken == Self.speechEndID {
                    vaeState = vaeDecoder.makeState()
                    semState = semanticEncoder.makeState()
                }
                nextEmbed = embedTokens.lookup([nextToken])
            }

            // LM step (zero-copy argmax)
            let (h, tok) = try decodeLMStep(
                model: lm, state: state,
                hidden: nextEmbed, position: position,
                hiddenSize: hs, headDim: hd, vocabSize: arch.vocabSize
            )
            currentHidden = Array(h[0 ..< hs])
            nextToken = tok
            position += 1
        }

    }

    private func argmax(_ data: [Float], offset: Int, count: Int) -> Int {
        var maxIdx = 0
        var maxVal = data[offset]
        for i in 1 ..< count {
            if data[offset + i] > maxVal {
                maxVal = data[offset + i]
                maxIdx = i
            }
        }
        return maxIdx
    }
}
