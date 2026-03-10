import Foundation

/// Numpy-compatible Mersenne Twister (MT19937) random number generator.
/// Produces identical sequences to `numpy.random.RandomState(seed)`.
struct NumpyRNG {
    private var mt = [UInt32](repeating: 0, count: 624)
    private var index: Int = 624
    private var hasSpare = false
    private var spare: Double = 0.0

    init(seed: UInt64) {
        mt[0] = UInt32(seed & 0xFFFF_FFFF)
        for i in 1 ..< 624 {
            mt[i] = 1812433253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
    }

    private mutating func generateNumbers() {
        for i in 0 ..< 624 {
            let y = (mt[i] & 0x8000_0000) | (mt[(i + 1) % 624] & 0x7FFF_FFFF)
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1 != 0 {
                mt[i] ^= 0x9908_B0DF
            }
        }
        index = 0
    }

    /// Generate a random UInt32 (genrand_int32).
    mutating func nextUInt32() -> UInt32 {
        if index >= 624 {
            generateNumbers()
        }
        var y = mt[index]
        index += 1

        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        return y
    }

    /// Generate a random integer in [0, bound) — matches numpy's randint(0, bound).
    /// Uses rejection sampling for unbiased results (same as numpy).
    mutating func randint(_ bound: UInt32) -> UInt32 {
        if bound == 0 { return 0 }
        // For power-of-2 bounds, simple masking works
        if bound & (bound - 1) == 0 {
            return nextUInt32() & (bound - 1)
        }
        // Rejection sampling
        let limit = UInt32.max - (UInt32.max % bound)
        var r: UInt32
        repeat {
            r = nextUInt32()
        } while r >= limit
        return r % bound
    }

    /// Generate a standard normal double using numpy's algorithm.
    /// Numpy's legacy RandomState uses the "standard_normal" from MT19937
    /// which is based on the Kinderman-Monahan ratio method via the Box-Muller
    /// transform cached pair approach.
    private mutating func nextDouble() -> Double {
        // numpy: (genrand_int32() >> 5) * (1.0/67108864.0) gives upper bits,
        // then (genrand_int32() >> 6) * (1.0/67108864.0)... actually numpy uses:
        // double = (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
        // where a = genrand_int32() >> 5, b = genrand_int32() >> 6
        let a = Double(nextUInt32() >> 5)
        let b = Double(nextUInt32() >> 6)
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// Generate a standard normal Float using numpy's Gauss (Box-Muller) method.
    /// This matches numpy.random.RandomState.standard_normal() / randn().
    mutating func nextGauss() -> Float {
        if hasSpare {
            hasSpare = false
            return Float(spare)
        }

        // Numpy's legacy RandomState uses the polar Box-Muller transform
        var x1, x2, r2: Double
        repeat {
            x1 = 2.0 * nextDouble() - 1.0
            x2 = 2.0 * nextDouble() - 1.0
            r2 = x1 * x1 + x2 * x2
        } while r2 >= 1.0 || r2 == 0.0

        let f = sqrt(-2.0 * log(r2) / r2)
        spare = f * x1
        hasSpare = true
        return Float(f * x2)
    }

    /// Generate array of normal-distributed Floats matching numpy's randn(1, count).
    mutating func randn(_ count: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            result[i] = nextGauss()
        }
        return result
    }
}
