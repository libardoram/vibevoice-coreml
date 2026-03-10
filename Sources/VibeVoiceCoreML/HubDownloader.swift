import Foundation
import HuggingFace

/// Downloads model files from Hugging Face Hub repositories.
/// Uses `swift-huggingface` for Python-compatible caching, Xet transport, and resumable downloads.
public enum HubDownloader {
    /// Format byte count as human-readable string.
    public static func formatBytes(_ bytes: Int) -> String {
        if bytes < 1024 { return "\(bytes) B" }
        let kb = Double(bytes) / 1024
        if kb < 1024 { return String(format: "%.0f KB", kb) }
        let mb = kb / 1024
        if mb < 1024 { return String(format: "%.0f MB", mb) }
        let gb = mb / 1024
        return String(format: "%.1f GB", gb)
    }

    /// Download a HuggingFace repository to local cache.
    /// Returns the local snapshot directory. Uses Python-compatible cache layout:
    ///   `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{commit}/`
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g. "gafiatulin/vibevoice-tts-0.5b-coreml")
    ///   - cacheDir: Override HF hub cache directory (default: auto-detected by HuggingFace library)
    ///   - maxConcurrent: Maximum number of parallel file downloads (default: 4)
    ///   - onProgress: Called with overall download progress (fractionCompleted 0.0–1.0).
    ///   - log: Optional verbose logging callback.
    public static func download(
        repo: String,
        to cacheDir: URL? = nil,
        maxConcurrent: Int = 4,
        onProgress: (@Sendable (_ path: String, _ size: Int, _ index: Int, _ total: Int, _ skipped: Bool) -> Void)? = nil,
        log: (@Sendable (_ message: String) -> Void)? = nil
    ) async throws -> URL {
        let client: HubClient
        if let cacheDir {
            let cache = HubCache(cacheDirectory: cacheDir)
            client = HubClient(cache: cache)
        } else {
            client = HubClient.default
        }

        log?("Downloading \(repo) (concurrency=\(maxConcurrent))...")

        // Progress only updates per-file (no partial byte updates from HF library)
        nonisolated(unsafe) var lastCompleted: Int64 = 0
        let handler: (@MainActor @Sendable (Progress) -> Void)? = if log != nil {
            { @MainActor @Sendable progress in
                let completed = progress.completedUnitCount
                guard completed != lastCompleted else { return }
                lastCompleted = completed
                let pct = Int(progress.fractionCompleted * 100)
                let done = formatBytes(Int(completed))
                let total = formatBytes(Int(progress.totalUnitCount))
                print("\r  Downloading... \(pct)% (\(done) / \(total))   ", terminator: "")
                fflush(stdout)
                if pct >= 100 { print() }
            }
        } else {
            nil
        }

        guard let repoID = Repo.ID(rawValue: repo) else {
            throw URLError(.badURL, userInfo: [NSLocalizedDescriptionKey: "Invalid repo ID: \(repo)"])
        }

        let snapshotDir = try await client.downloadSnapshot(
            of: repoID,
            matching: ["*.mlmodelc/*.*", "*.mlpackage/*.*", "*.bin", "*.vvvoice",
                        "voices/*.*", "tokenizer.json", "tokenizer_config.json"],
            maxConcurrentDownloads: maxConcurrent,
            progressHandler: handler
        )

        log?("Cache: \(snapshotDir.path)")
        return snapshotDir
    }
}
