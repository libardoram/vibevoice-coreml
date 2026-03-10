import Foundation
import Tokenizers

enum TokenizerLoader {
    /// Load tokenizer from a directory containing tokenizer.json.
    static func load(from directory: URL) async throws -> any Tokenizer {
        try await AutoTokenizer.from(modelFolder: directory)
    }
}
