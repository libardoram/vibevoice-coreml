import Foundation

/// Compute Rotary Position Embeddings for a sequence of positions.
/// Returns (cos, sin) arrays each of shape (Q * headDim) flattened.
func computeRoPE(positions: [Int], headDim: Int = Constants.headDim) -> (cos: [Float], sin: [Float]) {
    let halfDim = headDim / 2
    let q = positions.count

    // inv_freq = 1 / (theta ^ (2i / headDim)) for i in 0..<halfDim
    var invFreq = [Double](repeating: 0, count: halfDim)
    for i in 0 ..< halfDim {
        let exp = Double(2 * i) / Double(headDim)
        invFreq[i] = 1.0 / pow(Constants.ropeTheta, exp)
    }

    // For each position, compute freqs and duplicate
    var cosValues = [Float](repeating: 0, count: q * headDim)
    var sinValues = [Float](repeating: 0, count: q * headDim)

    for (pi, pos) in positions.enumerated() {
        let p = Double(pos)
        let base = pi * headDim
        for j in 0 ..< halfDim {
            let freq = p * invFreq[j]
            let c = Float(cos(freq))
            let s = Float(sin(freq))
            cosValues[base + j] = c
            cosValues[base + halfDim + j] = c  // duplicate
            sinValues[base + j] = s
            sinValues[base + halfDim + j] = s
        }
    }

    return (cosValues, sinValues)
}

/// Single-position RoPE (convenience).
func computeRoPE(position: Int, headDim: Int = Constants.headDim) -> (cos: [Float], sin: [Float]) {
    computeRoPE(positions: [position], headDim: headDim)
}
