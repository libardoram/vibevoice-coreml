"""Convert 0.5B voice prompts from .pt (PyTorch DynamicCache) to .vvvoice (flat binary).

Binary format (.vvvoice):
  Header (8 bytes):
    magic:        4 bytes  "VVVP"
    version:      uint16   = 1
    num_sections: uint16   = 4

  Section table (4 × 16 bytes = 64 bytes):
    For each section (lm, tts_lm, neg_lm, neg_tts_lm):
      num_layers:  uint16
      num_kv_heads: uint16
      seq_len:     uint32
      head_dim:    uint16
      hidden_dim:  uint16
      data_offset: uint32   (byte offset from file start)

  Data (contiguous float16):
    For each section:
      k_cache:     float16[num_layers * num_kv_heads * seq_len * head_dim]
      v_cache:     float16[num_layers * num_kv_heads * seq_len * head_dim]
      last_hidden: float16[hidden_dim]  (last token only)

Usage:
    uv run python convert/convert_voices.py [--voice-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"VVVP"
VERSION = 1
SECTION_NAMES = ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]
HEADER_SIZE = 8
SECTION_ENTRY_SIZE = 16  # 2+2+4+2+2+4 = 16 bytes


def extract_section(prompt_section: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract KV caches and last hidden state from one prompt section.

    Returns:
        k_cache: float16 array, shape (num_layers * num_kv_heads, seq_len, head_dim)
        v_cache: float16 array, same shape
        last_hidden: float16 array, shape (hidden_dim,)
        meta: dict with num_layers, num_kv_heads, seq_len, head_dim, hidden_dim
    """
    pkv = prompt_section["past_key_values"]
    hidden = prompt_section["last_hidden_state"]

    # DynamicCache stores key_cache / value_cache as lists of tensors
    k_list = pkv.key_cache   # list of (1, num_kv_heads, seq_len, head_dim)
    v_list = pkv.value_cache

    num_layers = len(k_list)
    _, num_kv_heads, seq_len, head_dim = k_list[0].shape
    hidden_dim = hidden.shape[-1]

    # Concatenate across layers → (1, total_kv, seq_len, head_dim), then drop batch dim
    k_cache = torch.cat(k_list, dim=1)[0].float().numpy().astype(np.float16)
    v_cache = torch.cat(v_list, dim=1)[0].float().numpy().astype(np.float16)

    # Only keep last token of hidden state
    last_hidden = hidden[0, -1, :].float().numpy().astype(np.float16)

    meta = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
    }
    return k_cache, v_cache, last_hidden, meta


def write_vvvoice(sections: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]], output_path: Path):
    """Write sections to .vvvoice binary file."""
    num_sections = len(sections)
    table_size = num_sections * SECTION_ENTRY_SIZE
    data_start = HEADER_SIZE + table_size

    # Compute data offsets
    offset = data_start
    offsets = []
    for k_cache, v_cache, last_hidden, meta in sections:
        offsets.append(offset)
        offset += k_cache.nbytes + v_cache.nbytes + last_hidden.nbytes

    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<HH", VERSION, num_sections))

        # Section table
        for i, (k_cache, v_cache, last_hidden, meta) in enumerate(sections):
            f.write(struct.pack(
                "<HHIHHI",
                meta["num_layers"],
                meta["num_kv_heads"],
                meta["seq_len"],
                meta["head_dim"],
                meta["hidden_dim"],
                offsets[i],
            ))

        # Data
        for k_cache, v_cache, last_hidden, meta in sections:
            f.write(k_cache.tobytes())
            f.write(v_cache.tobytes())
            f.write(last_hidden.tobytes())


def read_vvvoice(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """Read .vvvoice binary file back into sections (for verification)."""
    with open(path, "rb") as f:
        data = f.read()

    magic = data[:4]
    assert magic == MAGIC, f"Bad magic: {magic}"
    version, num_sections = struct.unpack_from("<HH", data, 4)
    assert version == VERSION, f"Unsupported version: {version}"

    sections = []
    for i in range(num_sections):
        entry_offset = HEADER_SIZE + i * SECTION_ENTRY_SIZE
        num_layers, num_kv_heads, seq_len, head_dim, hidden_dim, data_offset = struct.unpack_from(
            "<HHIHHI", data, entry_offset
        )

        total_kv = num_layers * num_kv_heads
        kv_elems = total_kv * seq_len * head_dim
        kv_bytes = kv_elems * 2  # float16

        k_cache = np.frombuffer(data, dtype=np.float16, count=kv_elems, offset=data_offset)
        k_cache = k_cache.reshape(total_kv, seq_len, head_dim)

        v_cache = np.frombuffer(data, dtype=np.float16, count=kv_elems, offset=data_offset + kv_bytes)
        v_cache = v_cache.reshape(total_kv, seq_len, head_dim)

        last_hidden = np.frombuffer(data, dtype=np.float16, count=hidden_dim, offset=data_offset + 2 * kv_bytes)

        meta = {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "hidden_dim": hidden_dim,
        }
        sections.append((k_cache.copy(), v_cache.copy(), last_hidden.copy(), meta))

    return sections


def convert_voice(pt_path: Path, output_dir: Path) -> Path:
    """Convert a single .pt voice prompt to .vvvoice."""
    prompt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    sections = []
    for name in SECTION_NAMES:
        k, v, h, meta = extract_section(prompt[name])
        sections.append((k, v, h, meta))

    output_path = output_dir / f"{pt_path.stem}.vvvoice"
    write_vvvoice(sections, output_path)
    return output_path


def verify_voice(pt_path: Path, vv_path: Path) -> bool:
    """Verify .vvvoice matches original .pt exactly."""
    prompt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sections = read_vvvoice(vv_path)

    all_ok = True
    for i, name in enumerate(SECTION_NAMES):
        k_orig, v_orig, h_orig, meta_orig = extract_section(prompt[name])
        k_read, v_read, h_read, meta_read = sections[i]

        if meta_orig != meta_read:
            print(f"  FAIL {name}: meta mismatch {meta_orig} vs {meta_read}")
            all_ok = False
            continue

        k_match = np.array_equal(k_orig, k_read)
        v_match = np.array_equal(v_orig, v_read)
        h_match = np.array_equal(h_orig, h_read)

        if not (k_match and v_match and h_match):
            print(f"  FAIL {name}: k={k_match} v={v_match} h={h_match}")
            all_ok = False
        else:
            s = meta_read["seq_len"]
            print(f"  OK   {name}: {meta_read['num_layers']}L × {meta_read['num_kv_heads']}KV × {s}seq × {meta_read['head_dim']}d")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Convert 0.5B voice prompts .pt → .vvvoice")
    parser.add_argument("--voice-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "voices",
                        help="Directory containing .pt voice prompts")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as voice-dir)")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify converted files (default: True)")
    parser.add_argument("--no-verify", action="store_false", dest="verify")
    args = parser.parse_args()

    voice_dir = args.voice_dir
    output_dir = args.output_dir or voice_dir

    pt_files = sorted(voice_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {voice_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(pt_files)} voice prompts → .vvvoice")
    print(f"  Source: {voice_dir}")
    print(f"  Output: {output_dir}")
    print()

    for pt_path in pt_files:
        pt_size = pt_path.stat().st_size
        vv_path = convert_voice(pt_path, output_dir)
        vv_size = vv_path.stat().st_size
        ratio = vv_size / pt_size * 100

        print(f"{pt_path.stem}:")
        print(f"  {pt_size / 1024:.0f}KB → {vv_size / 1024:.0f}KB ({ratio:.0f}%)")

        if args.verify:
            ok = verify_voice(pt_path, vv_path)
            if not ok:
                print(f"  VERIFICATION FAILED!")
                return

        print()

    total_pt = sum(f.stat().st_size for f in pt_files)
    total_vv = sum((output_dir / f"{f.stem}.vvvoice").stat().st_size for f in pt_files)
    print(f"Total: {total_pt / 1024 / 1024:.1f}MB → {total_vv / 1024 / 1024:.1f}MB ({total_vv / total_pt * 100:.0f}%)")
    print("All voices converted and verified OK." if args.verify else "All voices converted.")


if __name__ == "__main__":
    main()
