import Foundation

enum Constants {
    static let hiddenSize = 896
    static let headDim = 64
    static let numQHeads = 14
    static let numKVHeads = 2
    static let baseLMLayers = 4
    static let ttsLMLayers = 20
    static let vocabSize = 151936

    static let textWindowSize = 5
    static let speechWindowSize = 6

    static let sampleRate = 24000
    static let samplesPerFrame = 3200
    static let vaeDim = 64

    static let speechScaling: Float = 0.23339844
    static let speechBias: Float = -0.0703125

    static let ropeTheta: Double = 1e6
    static let maskValue: Float = -1e9
}
