import CoreML
import Foundation

// MARK: - MLMultiArray creation helpers

func makeMultiArray(_ data: [Float], shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    data.withUnsafeBufferPointer { src in
        ptr.update(from: src.baseAddress!, count: src.count)
    }
    return arr
}

func makeZeroArray(shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    let count = shape.reduce(1, *)
    ptr.update(repeating: 0, count: count)
    return arr
}

func readMultiArray(_ arr: MLMultiArray) -> [Float] {
    let count = arr.count
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

/// Write Float data into an existing MLMultiArray (reuse allocation).
func fillMultiArray(_ arr: MLMultiArray, with data: [Float]) {
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    data.withUnsafeBufferPointer { src in
        ptr.update(from: src.baseAddress!, count: src.count)
    }
}

/// Argmax directly on MLMultiArray without copying — for logits with shape (1, seqLen, vocabSize).
/// Returns argmax of the last token's logits.
func argmaxMultiArray(_ arr: MLMultiArray, lastTokenOffset: Int, count: Int) -> Int {
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    let base = ptr + lastTokenOffset
    var maxIdx = 0
    var maxVal = base[0]
    for i in 1 ..< count {
        let v = base[i]
        if v > maxVal {
            maxVal = v
            maxIdx = i
        }
    }
    return maxIdx
}

// MARK: - Causal attention mask

/// Build causal attention mask: shape (1, 1, Q, totalLen).
/// Position i can attend to [0, startPos + i].
func buildCausalMask(q: Int, startPos: Int) throws -> MLMultiArray {
    let totalLen = startPos + q
    let arr = try MLMultiArray(shape: [1, 1, q, totalLen].map { NSNumber(value: $0) }, dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    let stride = totalLen

    for i in 0 ..< q {
        let rowOffset = i * stride
        let allowedEnd = startPos + i + 1
        for j in 0 ..< totalLen {
            ptr[rowOffset + j] = j < allowedEnd ? 0.0 : Constants.maskValue
        }
    }
    return arr
}

// MARK: - LM forward pass

/// Forward pass through a stateful LM (base or TTS).
/// Returns flat array of (q * hiddenSize) floats.
func forwardLM(
    model: MLModel, state: MLState,
    hidden: [Float], pos: Int, q: Int,
    totalKV: Int
) throws -> [Float] {
    let positions = Array(pos ..< pos + q)
    let (cosArr, sinArr) = computeRoPE(positions: positions)

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_states": makeMultiArray(hidden, shape: [1, q, Constants.hiddenSize]),
        "position_cos": makeMultiArray(cosArr, shape: [1, q, Constants.headDim]),
        "position_sin": makeMultiArray(sinArr, shape: [1, q, Constants.headDim]),
        "attention_mask": buildCausalMask(q: q, startPos: pos),
        "inject_mode": makeMultiArray([Float(0)], shape: [1]),
        "inject_k": makeZeroArray(shape: [1, totalKV, q, Constants.headDim]),
        "inject_v": makeZeroArray(shape: [1, totalKV, q, Constants.headDim]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input, using: state)
    guard let outputHidden = output.featureValue(for: "output_hidden")?.multiArrayValue else {
        throw VibeVoiceError.modelError("Missing output_hidden from LM")
    }
    return readMultiArray(outputHidden)
}

// MARK: - KV Cache injection

/// Inject pre-computed KV cache into a stateful model.
/// Returns the sequence length (new position pointer).
func injectKVCache(
    model: MLModel, state: MLState,
    kCache: [Float], vCache: [Float],
    meta: VoicePromptMeta
) throws -> Int {
    let seqLen = meta.seqLen
    let totalKV = meta.numLayers * meta.numKVHeads

    let positions = Array(0 ..< seqLen)
    let (cosArr, sinArr) = computeRoPE(positions: positions)

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_states": makeZeroArray(shape: [1, seqLen, Constants.hiddenSize]),
        "position_cos": makeMultiArray(cosArr, shape: [1, seqLen, Constants.headDim]),
        "position_sin": makeMultiArray(sinArr, shape: [1, seqLen, Constants.headDim]),
        "attention_mask": buildCausalMask(q: seqLen, startPos: 0),
        "inject_mode": makeMultiArray([Float(1)], shape: [1]),
        "inject_k": makeMultiArray(kCache, shape: [1, totalKV, seqLen, meta.headDim]),
        "inject_v": makeMultiArray(vCache, shape: [1, totalKV, seqLen, meta.headDim]),
    ] as [String: MLMultiArray])

    let _ = try model.prediction(from: input, using: state)
    return seqLen
}

// MARK: - Diffusion (batched CFG, B=2)

/// Run diffusion with classifier-free guidance using B=2 batched model.
/// Returns predicted_v after CFG combination: v_uncond + scale * (v_cond - v_uncond).
func dpmSolverSample(
    model: MLModel,
    posCondition: [Float],
    negCondition: [Float],
    cfgScale: Float,
    schedule: DiffusionSchedule,
    numSteps: Int,
    rng: inout SeededRNG
) throws -> [Float] {
    let dim = Constants.vaeDim
    let hs = Constants.hiddenSize

    // Build the CFG-wrapped diffusion function
    let guidedFn: ([Float], Float, [Float]) throws -> [Float] = { sample, timestep, condition in
        // Batch: [positive, negative]
        var sampleB2 = [Float](repeating: 0, count: 2 * dim)
        var timestepB2 = [Float](repeating: 0, count: 2)
        var condB2 = [Float](repeating: 0, count: 2 * hs)

        for j in 0 ..< dim { sampleB2[j] = sample[j]; sampleB2[dim + j] = sample[j] }
        timestepB2[0] = timestep; timestepB2[1] = timestep
        for j in 0 ..< hs { condB2[j] = condition[j]; condB2[hs + j] = negCondition[j] }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "noisy_latent": makeMultiArray(sampleB2, shape: [2, dim]),
            "timestep": makeMultiArray(timestepB2, shape: [2]),
            "condition": makeMultiArray(condB2, shape: [2, hs]),
        ] as [String: MLMultiArray])

        let output = try model.prediction(from: input)
        guard let vB2 = output.featureValue(for: "predicted_noise")?.multiArrayValue else {
            throw VibeVoiceError.modelError("Missing predicted_noise from diffusion")
        }
        let vAll = readMultiArray(vB2)

        // CFG: v = v_uncond + scale * (v_cond - v_uncond)
        var result = [Float](repeating: 0, count: dim)
        for j in 0 ..< dim {
            let vCond = vAll[j]
            let vUncond = vAll[dim + j]
            result[j] = vUncond + cfgScale * (vCond - vUncond)
        }
        return result
    }

    return try VibeVoiceCoreML.dpmSolverSample(
        diffusionFn: guidedFn,
        condition: posCondition,
        numSteps: numSteps,
        schedule: schedule,
        rng: &rng
    )
}

// MARK: - VAE Decoder

/// Decode a single latent frame through the streaming VAE decoder.
/// Input: latent (vaeDim,), Output: audio samples (samplesPerFrame,).
func decodeVAE(model: MLModel, state: MLState, latent: [Float]) throws -> [Float] {
    // Transpose: (vaeDim,) → (1, vaeDim, 1) — channels-first for conv
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "latent": makeMultiArray(latent, shape: [1, Constants.vaeDim, 1]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input, using: state)
    guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
        throw VibeVoiceError.modelError("Missing audio from VAE decoder")
    }
    return readMultiArray(audio)
}

// MARK: - Acoustic Connector

/// Project speech latent to LM embedding space.
/// Input: latent (vaeDim,), Output: embedding (hiddenSize,).
func forwardConnector(model: MLModel, latent: [Float]) throws -> [Float] {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "speech_latent": makeMultiArray(latent, shape: [1, 1, Constants.vaeDim]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input)
    guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
        throw VibeVoiceError.modelError("Missing embedding from connector")
    }
    return readMultiArray(embedding)
}

// MARK: - EOS Classifier

/// Check end-of-speech probability.
/// Input: hidden state (hiddenSize,), Output: probability [0, 1].
func checkEOS(model: MLModel, hidden: [Float]) throws -> Float {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_state": makeMultiArray(hidden, shape: [1, Constants.hiddenSize]),
    ] as [String: MLMultiArray])

    let output = try model.prediction(from: input)
    guard let prob = output.featureValue(for: "eos_probability")?.multiArrayValue else {
        throw VibeVoiceError.modelError("Missing eos_probability")
    }
    return readMultiArray(prob)[0]
}
