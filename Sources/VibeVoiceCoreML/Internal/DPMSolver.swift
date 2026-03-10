import Foundation

/// Precomputed cosine noise schedule for DDPM (1000 steps).
struct DiffusionSchedule {
    let alpha: [Float]   // sqrt(alphas_cumprod), len=1000
    let sigma: [Float]   // sqrt(1 - alphas_cumprod), len=1000
    let lambda: [Float]  // log(alpha / sigma), len=1000

    init() {
        let n = 1000
        var ac = [Double](repeating: 0, count: n)
        for i in 0 ..< n {
            let s = Double(i) / Double(n)
            let val = cos((s + 0.008) / 1.008 * .pi / 2.0)
            ac[i] = val * val
        }
        let ac0 = ac[0]
        for i in 0 ..< n {
            ac[i] /= ac0
        }

        var a = [Float](repeating: 0, count: n)
        var s = [Float](repeating: 0, count: n)
        var l = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            a[i] = Float(sqrt(ac[i]))
            s[i] = Float(sqrt(1.0 - ac[i]))
            l[i] = Float(log(sqrt(ac[i]) / max(sqrt(1.0 - ac[i]), 1e-10)))
        }

        alpha = a
        sigma = s
        lambda = l
    }

    /// Create evenly-spaced timestep schedule from 999 down to 0.
    func makeTimesteps(numSteps: Int) -> [Int] {
        var ts = [Int](repeating: 0, count: numSteps + 1)
        for i in 0 ... numSteps {
            ts[i] = Int(round(Double(999) * Double(numSteps - i) / Double(numSteps)))
        }
        return ts
    }
}

/// Seeded random number generator (xoshiro256**).
struct SeededRNG {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // SplitMix64 to initialize state
        var s = seed
        func next() -> UInt64 {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        state = (next(), next(), next(), next())
    }

    mutating func nextUInt64() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    /// Generate a normal-distributed Float using Box-Muller transform.
    mutating func nextNormal() -> Float {
        let u1 = max(Float(nextUInt64()) / Float(UInt64.max), Float.leastNormalMagnitude)
        let u2 = Float(nextUInt64()) / Float(UInt64.max)
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }

    /// Generate array of normal-distributed Floats.
    mutating func randn(_ count: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            result[i] = nextNormal()
        }
        return result
    }
}

private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
    (x << k) | (x >> (64 - k))
}

/// DPM-Solver++ 2M sampler with classifier-free guidance.
///
/// - Parameters:
///   - diffusionFn: (sample [vaeDim], timestep, condition [hiddenSize]) → predicted_v [vaeDim]
///   - condition: Positive conditioning vector (hiddenSize)
///   - numSteps: Number of solver steps
///   - schedule: Precomputed noise schedule
///   - rng: Seeded RNG for initial noise
/// - Returns: Denoised latent of shape (vaeDim,)
func dpmSolverSample(
    diffusionFn: ([Float], Float, [Float]) throws -> [Float],
    condition: [Float],
    numSteps: Int,
    schedule: DiffusionSchedule,
    rng: inout SeededRNG
) rethrows -> [Float] {
    let dim = Constants.vaeDim
    let tSchedule = schedule.makeTimesteps(numSteps: numSteps)

    var sample = rng.randn(dim)
    var x0List: [[Float]] = []

    for i in 0 ..< numSteps {
        let s = tSchedule[i]
        let t = tSchedule[i + 1]

        // v-prediction
        let v = try diffusionFn(sample, Float(s), condition)

        // Predict x0: x0 = alpha[s] * sample - sigma[s] * v
        let alphaS = schedule.alpha[s]
        let sigmaS = schedule.sigma[s]
        var x0 = [Float](repeating: 0, count: dim)
        for j in 0 ..< dim {
            x0[j] = alphaS * sample[j] - sigmaS * v[j]
        }
        x0List.append(x0)

        let lamS = schedule.lambda[s]
        let lamT = schedule.lambda[max(t, 0)]
        let h = lamT - lamS

        // Determine solver order
        let isLast = i == numSteps - 1
        let isSecondLast = i == numSteps - 2
        let lowerOrderFinal = isLast && numSteps < 15
        let lowerOrderSecond = isSecondLast && numSteps < 15
        let useFirstOrder = x0List.count < 2 || lowerOrderFinal || lowerOrderSecond

        // Compute D (direction)
        var d = [Float](repeating: 0, count: dim)
        if useFirstOrder {
            d = x0List.last!
        } else {
            let sPrev = tSchedule[i - 1]
            let lamSPrev = schedule.lambda[sPrev]
            let hPrev = lamS - lamSPrev
            let r = hPrev / h
            let d0 = x0List[x0List.count - 1]
            let d_prev = x0List[x0List.count - 2]
            for j in 0 ..< dim {
                let d1 = (1.0 / r) * (d0[j] - d_prev[j])
                d[j] = d0[j] + 0.5 * d1
            }
        }

        // Update sample
        let sigmaT = schedule.sigma[max(t, 0)]
        let alphaT = schedule.alpha[max(t, 0)]
        let expm1H = expm1f(Float(-h))
        for j in 0 ..< dim {
            sample[j] = (sigmaT / sigmaS) * sample[j] - alphaT * expm1H * d[j]
        }
    }

    return sample
}
