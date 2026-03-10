#!/usr/bin/env python3
"""Build HuggingFace distribution repos from converted CoreML models.

Assembles staging directories ready for `huggingface-cli upload`. Optionally
compiles .mlpackage → .mlmodelc (requires Xcode with coremlcompiler).

Usage:
    python build_hf_repos.py [--compile] [--staging-dir DIR] [REPO...]

    # Stage all repos (copy .mlpackage as-is):
    python build_hf_repos.py

    # Stage specific repos:
    python build_hf_repos.py tts-0.5b tts-7b

    # Compile to .mlmodelc (requires Xcode):
    python build_hf_repos.py --compile

    # Regenerate missing embedding .bin files from HF cache:
    python build_hf_repos.py --regen-embeddings

    # Upload after staging:
    hf upload gafiatulin/vibevoice-tts-0.5b-coreml dist/vibevoice-tts-0.5b-coreml/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGING = ROOT / "dist"

# HuggingFace cache for tokenizer
HF_CACHE = Path.home() / ".cache/huggingface/hub"


# ─── Repo manifests ──────────────────────────────────────────────────────────

@dataclass
class RepoManifest:
    name: str                         # HF repo suffix: .../{name}
    models: dict[str, Path]           # dest_name → source .mlpackage path
    binaries: dict[str, Path]         # dest_name → source .bin path
    extras: dict[str, Path] = field(default_factory=dict)  # other files
    description: str = ""


def _build_dir(subpath: str) -> Path:
    return ROOT / subpath


MANIFESTS: dict[str, RepoManifest] = {
    "asr": RepoManifest(
        name="vibevoice-asr-coreml",
        description="VibeVoice ASR (Qwen2-7B, 8.3B params) — CoreML INT8, fused LM+head, fused encoder, fused projector. 50+ languages, 60-minute single-pass transcription.",
        models={
            "fused_encoder": _build_dir("stt/build/vibevoice-asr/fused_encoder.mlpackage"),
            "fused_projector": _build_dir("stt/build/vibevoice-asr/fused_projector.mlpackage"),
            "lm_decoder_fused_int8": _build_dir("stt/build/vibevoice-asr/lm_decoder_fused_int8.mlpackage"),
        },
        binaries={
            "embed_tokens.bin": _build_dir("stt/build/vibevoice-asr/embed_tokens.bin"),
        },
    ),
    "tts-0.5b": RepoManifest(
        name="vibevoice-tts-0.5b-coreml",
        description="VibeVoice Realtime 0.5B (Qwen2.5-0.5B) — CoreML streaming TTS, 25 built-in voices across 10 languages.",
        models={
            "base_lm_stateful": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/base_lm_stateful.mlpackage"),
            "tts_lm_stateful": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/tts_lm_stateful.mlpackage"),
            "diffusion_head_b2": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/diffusion_head_b2.mlpackage"),
            "vae_decoder_streaming": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/vae_decoder_streaming.mlpackage"),
            "eos_classifier": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/eos_classifier.mlpackage"),
            "acoustic_connector": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/acoustic_connector.mlpackage"),
        },
        binaries={
            "embed_tokens.bin": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/embed_tokens.bin"),
            "tts_input_types.bin": _build_dir("tts/vibevoice-realtime-0.5b/build/vibevoice-realtime-0.5b/tts_input_types.bin"),
        },
    ),
    "tts-1.5b": RepoManifest(
        name="vibevoice-tts-1.5b-coreml",
        description="VibeVoice 1.5B (Qwen2.5-1.5B) — CoreML INT8, fused LM+head, fused diffusion loop, DPM-Solver++ 10-step. Multi-speaker TTS with voice cloning.",
        models={
            "lm_decoder_fused_int8": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/lm_decoder_fused_int8.mlpackage"),
            "diffusion_loop": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/diffusion_loop.mlpackage"),
            "vae_decoder_streaming": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/vae_decoder_streaming.mlpackage"),
            "semantic_encoder_streaming": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/semantic_encoder_streaming.mlpackage"),
            "acoustic_connector": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/acoustic_connector.mlpackage"),
            "semantic_connector": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/semantic_connector.mlpackage"),
            "vae_encoder": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/vae_encoder.mlpackage"),
        },
        binaries={
            "embed_tokens.bin": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-1.5b/embed_tokens.bin"),
        },
    ),
    "tts-7b": RepoManifest(
        name="vibevoice-tts-7b-coreml",
        description="VibeVoice 7B (Qwen2.5-7B) — CoreML INT8, fused LM+head, fused diffusion loop, DPM-Solver++ 10-step. Multi-speaker TTS with voice cloning.",
        models={
            "lm_decoder_fused_int8": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/lm_decoder_fused_int8.mlpackage"),
            "diffusion_loop": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/diffusion_loop.mlpackage"),
            "vae_decoder_streaming": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/vae_decoder_streaming.mlpackage"),
            "semantic_encoder_streaming": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/semantic_encoder_streaming.mlpackage"),
            "acoustic_connector": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/acoustic_connector.mlpackage"),
            "semantic_connector": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/semantic_connector.mlpackage"),
            "vae_encoder": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/vae_encoder.mlpackage"),
        },
        binaries={
            "embed_tokens.bin": _build_dir("tts/vibevoice-multispeaker/build/vibevoice-7b/embed_tokens.bin"),
        },
    ),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_tokenizer_files() -> dict[str, Path]:
    """Find tokenizer.json and tokenizer_config.json in HuggingFace cache."""
    candidates = [
        "models--microsoft--VibeVoice-ASR-HF",
        "models--Qwen--Qwen2.5-0.5B",
        "models--Qwen--Qwen2.5-1.5B",
        "models--Qwen--Qwen2.5-7B",
    ]
    result: dict[str, Path] = {}
    for model_dir_name in candidates:
        model_dir = HF_CACHE / model_dir_name
        if not model_dir.exists():
            continue
        for name in ["tokenizer.json", "tokenizer_config.json"]:
            if name not in result:
                for f in model_dir.rglob(name):
                    result[name] = f
                    break
        if len(result) == 2:
            break
    return result


def _xcrun_env() -> dict[str, str]:
    """Return env dict with DEVELOPER_DIR pointing to Xcode if needed."""
    import os
    env = os.environ.copy()
    # If xcrun can't find coremlcompiler with current config, try Xcode.app
    result = subprocess.run(
        ["xcrun", "-find", "coremlcompiler"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    if result.returncode == 0:
        return env
    xcode_dev = "/Applications/Xcode.app/Contents/Developer"
    if Path(xcode_dev).exists():
        env["DEVELOPER_DIR"] = xcode_dev
    return env


def has_coremlcompiler() -> bool:
    try:
        env = _xcrun_env()
        result = subprocess.run(
            ["xcrun", "-find", "coremlcompiler"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def compile_mlpackage(src: Path, dest_dir: Path) -> Path:
    """Compile .mlpackage → .mlmodelc using coremlcompiler."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    env = _xcrun_env()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(src), str(dest_dir)],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    if result.returncode != 0:
        print(f"  ERROR compiling {src.name}: {result.stderr.strip()}")
        sys.exit(1)
    compiled = dest_dir / f"{src.stem}.mlmodelc"
    if not compiled.exists():
        print(f"  ERROR: expected {compiled} not found after compilation")
        sys.exit(1)
    return compiled


def copy_tree(src: Path, dst: Path):
    """Copy file or directory tree."""
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def human_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ─── Embedding regeneration ───────────────────────────────────────────────────

# Map: repo key → (HF cache dir name, {bin_name: safetensors_key})
EMBEDDING_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "tts-0.5b": (
        "models--mlx-community--VibeVoice-Realtime-0.5B-fp16",
        {
            "embed_tokens.bin": "language_model.embed_tokens.weight",
            "tts_input_types.bin": "tts_input_types.weight",
        },
    ),
    "tts-1.5b": (
        "models--microsoft--VibeVoice-1.5B",
        {"embed_tokens.bin": "model.language_model.embed_tokens.weight"},
    ),
    "tts-7b": (
        "models--vibevoice--VibeVoice-7B",
        {"embed_tokens.bin": "model.language_model.embed_tokens.weight"},
    ),
}


def _find_safetensors(hf_dir_name: str) -> list[Path]:
    """Find all .safetensors files for a model in HF cache."""
    model_dir = HF_CACHE / hf_dir_name
    if not model_dir.exists():
        return []
    return sorted(model_dir.rglob("*.safetensors"))


def _write_embedding_bin(tensor, output_path: Path):
    """Write a tensor to .bin format: [shape[0]: uint32] [shape[1]: uint32] [data: float16]."""
    import numpy as np
    arr = tensor.float().numpy().astype(np.float16)
    with open(output_path, "wb") as f:
        f.write(np.array(list(arr.shape), dtype=np.uint32).tobytes())
        f.write(arr.tobytes())
    mb = output_path.stat().st_size / 1e6
    print(f"  Wrote {output_path.name}: {list(arr.shape)} ({mb:.1f}MB)")


def regen_embeddings(repo_keys: list[str]):
    """Regenerate missing embedding .bin files from HF cache safetensors."""
    import torch
    from safetensors import safe_open

    for key in repo_keys:
        if key not in EMBEDDING_SOURCES:
            continue

        manifest = MANIFESTS[key]
        hf_dir_name, key_map = EMBEDDING_SOURCES[key]

        # Check which bins are missing
        missing_bins = {
            bin_name: st_key
            for bin_name, st_key in key_map.items()
            if not manifest.binaries[bin_name].exists()
        }
        if not missing_bins:
            continue

        print(f"\nRegenerating embeddings for {manifest.name}...")
        safetensors_files = _find_safetensors(hf_dir_name)
        if not safetensors_files:
            print(f"  ERROR: HF cache not found for {hf_dir_name}")
            print(f"  Run: huggingface-cli download {hf_dir_name.replace('models--', '').replace('--', '/')}")
            continue

        # Search for keys across shards
        for bin_name, st_key in missing_bins.items():
            found = False
            for sf_path in safetensors_files:
                try:
                    f = safe_open(str(sf_path), framework="pt")
                    if st_key in f.keys():
                        tensor = f.get_tensor(st_key)
                        output_path = manifest.binaries[bin_name]
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        _write_embedding_bin(tensor, output_path)
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                print(f"  ERROR: key '{st_key}' not found in any safetensors shard")


# ─── Build logic ──────────────────────────────────────────────────────────────

def populate_extras():
    """Add tokenizer files to all repos, and voices to tts-0.5b."""
    tok_files = find_tokenizer_files()
    if not tok_files:
        print("WARNING: tokenizer files not found in HuggingFace cache")

    # Add tokenizer files to every repo
    for manifest in MANIFESTS.values():
        for name, path in tok_files.items():
            manifest.extras[name] = path

    # Add .vvvoice files to tts-0.5b as voices/ subdirectory
    voices_dir = ROOT / "tts/vibevoice-realtime-0.5b/voices"
    for vv in sorted(voices_dir.glob("*.vvvoice")):
        MANIFESTS["tts-0.5b"].extras[f"voices/{vv.name}"] = vv


def check_manifest(manifest: RepoManifest) -> list[str]:
    """Check all files in manifest exist. Returns list of missing files."""
    missing = []
    for name, path in manifest.models.items():
        if not path.exists():
            missing.append(f"model: {name} ({path})")
    for name, path in manifest.binaries.items():
        if not path.exists():
            missing.append(f"binary: {name} ({path})")
    for name, path in manifest.extras.items():
        if not path.exists():
            missing.append(f"extra: {name} ({path})")
    return missing


def stage_repo(manifest: RepoManifest, staging_dir: Path, compile: bool) -> Path:
    """Stage a single HF repo directory."""
    repo_dir = staging_dir / manifest.name
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)

    total_size = 0

    # Models
    for name, src in manifest.models.items():
        if compile:
            print(f"  Compiling {name}...")
            compiled = compile_mlpackage(src, repo_dir)
            total_size += dir_size(compiled)
        else:
            dst = repo_dir / src.name
            print(f"  Copying {name} ({human_size(dir_size(src))})...")
            copy_tree(src, dst)
            total_size += dir_size(dst)

    # Binaries
    for name, src in manifest.binaries.items():
        dst = repo_dir / name
        print(f"  Copying {name} ({human_size(src.stat().st_size)})...")
        shutil.copy2(src, dst)
        total_size += dst.stat().st_size

    # Extras
    for name, src in manifest.extras.items():
        dst = repo_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Copying {name} ({human_size(src.stat().st_size)})...")
        shutil.copy2(src, dst)
        total_size += dst.stat().st_size

    # Model card
    write_model_card(manifest, repo_dir, compile)

    return repo_dir


def write_model_card(manifest: RepoManifest, repo_dir: Path, compiled: bool):
    """Write a README.md model card for the HF repo."""
    ext = ".mlmodelc" if compiled else ".mlpackage"
    model_list = "\n".join(f"- `{name}{ext}`" for name in manifest.models)
    binary_list = "\n".join(f"- `{name}`" for name in manifest.binaries)
    extra_list = "\n".join(f"- `{name}`" for name in manifest.extras)

    files_section = ""
    if model_list:
        files_section += f"### Models\n{model_list}\n\n"
    if binary_list:
        files_section += f"### Data\n{binary_list}\n\n"
    if extra_list:
        files_section += f"### Extras\n{extra_list}\n\n"

    card = f"""---
license: mit
tags:
  - coreml
  - speech
  - {'tts' if 'tts' in manifest.name else 'asr' if 'asr' in manifest.name else 'voice-prompts'}
  - vibevoice
  - apple
---

# gafiatulin / {manifest.name}

{manifest.description}

## Requirements

- iOS 18+ / macOS 15+ (requires `ct.StateType` for stateful models)
- {'Pre-compiled `.mlmodelc` — no on-device compilation needed' if compiled else '`.mlpackage` format — compile with `coremlcompiler` or load directly'}

## Files

{files_section}## License

MIT (same as upstream VibeVoice models from Microsoft)
"""
    (repo_dir / "README.md").write_text(card)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build HuggingFace distribution repos")
    parser.add_argument("repos", nargs="*", default=list(MANIFESTS.keys()),
                        choices=list(MANIFESTS.keys()),
                        help="Which repos to build (default: all)")
    parser.add_argument("--compile", action="store_true",
                        help="Compile .mlpackage → .mlmodelc (requires Xcode)")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING,
                        help=f"Staging directory (default: {DEFAULT_STAGING})")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check for missing files, don't stage")
    parser.add_argument("--regen-embeddings", action="store_true",
                        help="Regenerate missing embed_tokens.bin / tts_input_types.bin from HF cache")
    parser.add_argument("--readme-only", action="store_true",
                        help="Only regenerate README.md in existing staging dirs")
    args = parser.parse_args()

    if args.compile and not has_coremlcompiler():
        print("ERROR: --compile requires Xcode with coremlcompiler")
        print("  Install Xcode from the App Store, then run:")
        print("  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer")
        sys.exit(1)

    # Populate dynamic extras (tokenizers, voices)
    populate_extras()

    # Regenerate missing embeddings if requested
    if args.regen_embeddings:
        regen_embeddings(args.repos)

    # Check all manifests
    print("Checking source files...\n")
    any_missing = False
    for key in args.repos:
        manifest = MANIFESTS[key]
        missing = check_manifest(manifest)
        status = "OK" if not missing else f"MISSING {len(missing)} file(s)"
        n_files = len(manifest.models) + len(manifest.binaries) + len(manifest.extras)
        print(f"  {manifest.name}: {n_files} files — {status}")
        for m in missing:
            print(f"    ! {m}")
            any_missing = True

    if args.check_only:
        sys.exit(1 if any_missing else 0)

    if args.readme_only:
        compiled = any((args.staging_dir / MANIFESTS[k].name / f"{n}.mlmodelc").exists()
                       for k in args.repos for n in MANIFESTS[k].models)
        for key in args.repos:
            manifest = MANIFESTS[key]
            repo_dir = args.staging_dir / manifest.name
            if not repo_dir.exists():
                print(f"Skipping {manifest.name} (no staging dir)")
                continue
            write_model_card(manifest, repo_dir, compiled)
            print(f"  {manifest.name}: README.md updated")
        return

    if any_missing:
        print("\nSome files are missing. Fix the issues above or run conversion scripts first.")
        print("Continue with available files? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            sys.exit(1)

    # Stage repos
    print(f"\nStaging to {args.staging_dir}/\n")
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    for key in args.repos:
        manifest = MANIFESTS[key]
        missing = check_manifest(manifest)
        if missing:
            print(f"Skipping {manifest.name} ({len(missing)} missing files)")
            continue

        print(f"Building {manifest.name}...")
        repo_dir = stage_repo(manifest, args.staging_dir, args.compile)
        total = dir_size(repo_dir)
        n_files = sum(1 for f in repo_dir.rglob("*") if f.is_file())
        print(f"  → {repo_dir} ({human_size(total)}, {n_files} files)\n")

    # Summary
    print("=" * 60)
    print("Staging complete. To upload:\n")
    for key in args.repos:
        manifest = MANIFESTS[key]
        repo_dir = args.staging_dir / manifest.name
        if repo_dir.exists():
            print(f"  hf upload gafiatulin/{manifest.name} {repo_dir}/")
    print()


if __name__ == "__main__":
    main()
