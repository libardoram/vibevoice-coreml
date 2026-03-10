"""Traceable wrappers for VibeVoice multi-speaker TTS components.

Imports shared wrappers from common and adds model-specific ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from traceable_common import (  # noqa: E402, F401
    TraceableStreamingVAEDecoder,
    TraceableDiffusionHead,
    TraceableDiffusionLoopCFG,
    TraceableAcousticConnector,
    TraceableSemanticConnector,
    TraceableLMHead,
)


class TraceableVAEEncoder(nn.Module):
    """Acoustic VAE encoder: audio (B, 1, samples) -> latent (B, vae_dim, T)."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encoder(audio)
