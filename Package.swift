// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "VibeVoiceCoreML",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "VibeVoiceCoreML", targets: ["VibeVoiceCoreML"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.2"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.8.0"),
    ],
    targets: [
        .target(
            name: "VibeVoiceCoreML",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
        .executableTarget(
            name: "vibevoice-cli",
            dependencies: ["VibeVoiceCoreML"]
        ),
    ]
)
