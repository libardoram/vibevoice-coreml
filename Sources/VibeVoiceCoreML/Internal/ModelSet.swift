import CoreML

struct ModelSet {
    let baseLM: MLModel
    let ttsLM: MLModel
    let diffusion: MLModel
    let vaeDecoder: MLModel
    let eosClassifier: MLModel
    let acousticConnector: MLModel
}
