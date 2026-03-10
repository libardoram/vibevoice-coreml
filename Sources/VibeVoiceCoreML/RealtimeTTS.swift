import CoreML
import Foundation

// MARK: - Public API

/// Configuration for TTS generation.
public struct TTSConfig: Sendable {
    /// Voice name (e.g., "Emma", "Davis"). Matches .vvvoice filename stem.
    public var voice: String
    /// Classifier-free guidance scale. Higher = more adherent to conditioning.
    public var cfgScale: Float
    /// Number of DPM-Solver++ diffusion steps per frame.
    public var diffusionSteps: Int
    /// Random seed for reproducible output. nil = random.
    public var seed: UInt64?

    public init(
        voice: String = "Emma",
        cfgScale: Float = 1.5,
        diffusionSteps: Int = 5,
        seed: UInt64? = nil
    ) {
        self.voice = voice
        self.cfgScale = cfgScale
        self.diffusionSteps = diffusionSteps
        self.seed = seed
    }
}

/// A single audio frame produced during streaming generation.
public struct AudioFrame: Sendable {
    /// Raw PCM float32 samples at 24 kHz.
    public let samples: [Float]
    /// Frame index (0-based).
    public let index: Int
}

/// Real-time streaming TTS using the 0.5B model.
public final class RealtimeTTS: @unchecked Sendable {
    private let models: ModelSet
    private let embedTokens: EmbeddingTable
    private let ttsInputTypes: EmbeddingTable
    private let voicesDir: URL
    private let tokenizerDir: URL

    /// Load models from a directory containing .mlmodelc files, .bin files, and voices.
    ///
    /// Expected layout:
    /// ```
    /// modelsDir/
    ///   base_lm_stateful.mlmodelc
    ///   tts_lm_stateful.mlmodelc
    ///   diffusion_head_b2.mlmodelc
    ///   vae_decoder_streaming.mlmodelc
    ///   eos_classifier.mlmodelc
    ///   acoustic_connector.mlmodelc
    ///   embed_tokens.bin
    ///   tts_input_types.bin
    /// voicesDir/
    ///   en-Emma_woman.vvvoice
    ///   ...
    /// tokenizerDir/
    ///   tokenizer.json
    /// ```
    public init(modelsDir: URL, voicesDir: URL, tokenizerDir: URL) throws {
        self.voicesDir = voicesDir
        self.tokenizerDir = tokenizerDir

        let config = MLModelConfiguration()
        let configCPUGPU = MLModelConfiguration()
        configCPUGPU.computeUnits = .cpuAndGPU

        func loadModel(_ name: String, config cfg: MLModelConfiguration) throws -> MLModel {
            // Try .mlmodelc first (pre-compiled), then .mlpackage
            let compiledURL = modelsDir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL, configuration: cfg)
            }
            let packageURL = modelsDir.appendingPathComponent("\(name).mlpackage")
            let compiledDir = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledDir, configuration: cfg)
        }

        models = try ModelSet(
            baseLM: loadModel("base_lm_stateful", config: config),
            ttsLM: loadModel("tts_lm_stateful", config: config),
            diffusion: loadModel("diffusion_head_b2", config: config),
            vaeDecoder: loadModel("vae_decoder_streaming", config: configCPUGPU),
            eosClassifier: loadModel("eos_classifier", config: config),
            acousticConnector: loadModel("acoustic_connector", config: config)
        )

        embedTokens = try EmbeddingTable(contentsOf: modelsDir.appendingPathComponent("embed_tokens.bin"))
        ttsInputTypes = try EmbeddingTable(contentsOf: modelsDir.appendingPathComponent("tts_input_types.bin"))
    }

    /// Generate speech from text, streaming audio frames as they're produced.
    public func speak(_ text: String, config: TTSConfig = TTSConfig()) -> AsyncThrowingStream<AudioFrame, Error> {
        let generator = self
        return AsyncThrowingStream { continuation in
            Task { @Sendable in
                do {
                    try await generator.generate(text: text, config: config, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Generate speech and return all audio at once.
    public func synthesize(_ text: String, config: TTSConfig = TTSConfig()) async throws -> [Float] {
        var allSamples: [Float] = []
        for try await frame in speak(text, config: config) {
            allSamples.append(contentsOf: frame.samples)
        }
        return allSamples
    }
}

// MARK: - Internal generation

extension RealtimeTTS {
    private func generate(
        text: String,
        config: TTSConfig,
        continuation: AsyncThrowingStream<AudioFrame, Error>.Continuation
    ) async throws {
        // Tokenize
        let tokenizer = try await TokenizerLoader.load(from: tokenizerDir)
        let textWithNewline = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let tokenIds = tokenizer.encode(text: textWithNewline)

        // Load voice prompt
        let voicePrompt = try VoicePrompt(contentsOf: findVoice(config.voice))

        // Create model states
        let baseState = models.baseLM.makeState()
        let ttsState = models.ttsLM.makeState()
        let negTtsState = models.ttsLM.makeState()
        let vaeState = models.vaeDecoder.makeState()

        // Inject KV caches from voice prompt
        let basePos = try injectKVCache(
            model: models.baseLM, state: baseState,
            kCache: voicePrompt.sections[0].kCache,
            vCache: voicePrompt.sections[0].vCache,
            meta: voicePrompt.sections[0].meta
        )
        let ttsPos = try injectKVCache(
            model: models.ttsLM, state: ttsState,
            kCache: voicePrompt.sections[1].kCache,
            vCache: voicePrompt.sections[1].vCache,
            meta: voicePrompt.sections[1].meta
        )
        let negTtsPos = try injectKVCache(
            model: models.ttsLM, state: negTtsState,
            kCache: voicePrompt.sections[3].kCache,
            vCache: voicePrompt.sections[3].vCache,
            meta: voicePrompt.sections[3].meta
        )

        // Initial conditions from voice prompt
        var ttsLastHidden = voicePrompt.sections[1].lastHidden  // (896,) from tts_lm
        var negTtsLastHidden = voicePrompt.sections[3].lastHidden  // (896,) from neg_tts_lm

        // Diffusion setup
        let schedule = DiffusionSchedule()
        var rng = SeededRNG(seed: config.seed ?? UInt64.random(in: 0 ..< UInt64.max))

        // Generation loop
        var currentBasePos = basePos
        var currentTtsPos = ttsPos
        var currentNegTtsPos = negTtsPos
        var frameIndex = 0
        let totalWindows = (tokenIds.count + Constants.textWindowSize - 1) / Constants.textWindowSize

        var windowIdx = 0
        let maxFrames = max(tokenIds.count * 5, 30)
        while true {
            // Process text window
            let start = windowIdx * Constants.textWindowSize
            let end = min(start + Constants.textWindowSize, tokenIds.count)
            let windowTokens = start < tokenIds.count ? Array(tokenIds[start ..< end]) : []

            if !windowTokens.isEmpty {
                let textEmbeds = embedTokens.lookup(windowTokens)  // (Q, 896)
                let q = windowTokens.count

                // Base LM forward
                let baseLMOutput = try forwardLM(
                    model: models.baseLM, state: baseState,
                    hidden: textEmbeds, pos: currentBasePos, q: q,
                    totalKV: Constants.baseLMLayers * Constants.numKVHeads
                )
                currentBasePos += q

                // TTS LM forward (add text type embedding)
                var ttsEmbeds = baseLMOutput
                for i in 0 ..< q {
                    let offset = i * Constants.hiddenSize
                    for j in 0 ..< Constants.hiddenSize {
                        ttsEmbeds[offset + j] += ttsInputTypes.data[Constants.hiddenSize + j]  // type[1] = text
                    }
                }

                let ttsOutput = try forwardLM(
                    model: models.ttsLM, state: ttsState,
                    hidden: ttsEmbeds, pos: currentTtsPos, q: q,
                    totalKV: Constants.ttsLMLayers * Constants.numKVHeads
                )
                currentTtsPos += q

                // Take last token hidden state
                let lastOffset = (q - 1) * Constants.hiddenSize
                ttsLastHidden = Array(ttsOutput[lastOffset ..< lastOffset + Constants.hiddenSize])
            }

            // Generate speech frames
            var finished = false
            for _ in 0 ..< Constants.speechWindowSize {
                // Diffusion sampling with CFG
                let speechLatent = try dpmSolverSample(
                    model: models.diffusion,
                    posCondition: ttsLastHidden,
                    negCondition: negTtsLastHidden,
                    cfgScale: config.cfgScale,
                    schedule: schedule,
                    numSteps: config.diffusionSteps,
                    rng: &rng
                )

                // Scale and decode via streaming VAE
                var scaledLatent = [Float](repeating: 0, count: Constants.vaeDim)
                for i in 0 ..< Constants.vaeDim {
                    scaledLatent[i] = speechLatent[i] / Constants.speechScaling - Constants.speechBias
                }

                let audioFrame = try decodeVAE(
                    model: models.vaeDecoder, state: vaeState,
                    latent: scaledLatent
                )

                continuation.yield(AudioFrame(samples: audioFrame, index: frameIndex))
                frameIndex += 1

                // Acoustic connector feedback
                let speechEmbed = try forwardConnector(
                    model: models.acousticConnector, latent: speechLatent
                )

                // TTS LM forward with speech token
                var ttsInput = speechEmbed
                for j in 0 ..< Constants.hiddenSize {
                    ttsInput[j] += ttsInputTypes.data[j]  // type[0] = speech
                }

                let ttsOut = try forwardLM(
                    model: models.ttsLM, state: ttsState,
                    hidden: ttsInput, pos: currentTtsPos, q: 1,
                    totalKV: Constants.ttsLMLayers * Constants.numKVHeads
                )
                currentTtsPos += 1
                ttsLastHidden = ttsOut

                // Negative TTS LM forward
                var negInput = speechEmbed
                for j in 0 ..< Constants.hiddenSize {
                    negInput[j] += ttsInputTypes.data[j]
                }

                let negTtsOut = try forwardLM(
                    model: models.ttsLM, state: negTtsState,
                    hidden: negInput, pos: currentNegTtsPos, q: 1,
                    totalKV: Constants.ttsLMLayers * Constants.numKVHeads
                )
                currentNegTtsPos += 1
                negTtsLastHidden = negTtsOut

                // EOS check (only after all text windows consumed)
                if windowIdx >= totalWindows {
                    let eosProb = try checkEOS(model: models.eosClassifier, hidden: ttsLastHidden)
                    if eosProb > 0.5 {
                        finished = true
                        break
                    }
                }
            }

            if finished { break }
            if frameIndex >= maxFrames { break }
            windowIdx += 1
        }
    }

    private func findVoice(_ name: String) throws -> URL {
        let files = try FileManager.default.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: nil)
        let vvFiles = files.filter { $0.pathExtension == "vvvoice" }

        // Match by name (case-insensitive)
        if let match = vvFiles.first(where: {
            $0.deletingPathExtension().lastPathComponent
                .split(separator: "-").last?
                .split(separator: "_").first?
                .lowercased() == name.lowercased()
        }) {
            return match
        }

        // Fallback to first available
        guard let first = vvFiles.first else {
            throw VibeVoiceError.voiceNotFound(name)
        }
        return first
    }
}

// MARK: - Errors

public enum VibeVoiceError: Error, LocalizedError {
    case voiceNotFound(String)
    case invalidData(String)
    case modelError(String)

    public var errorDescription: String? {
        switch self {
        case .voiceNotFound(let name): return "Voice '\(name)' not found"
        case .invalidData(let msg): return "Invalid data: \(msg)"
        case .modelError(let msg): return "Model error: \(msg)"
        }
    }
}
