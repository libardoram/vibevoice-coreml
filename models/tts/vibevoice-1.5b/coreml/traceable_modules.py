"""Traceable wrappers for VibeVoice 1.5B/7B TTS components.

The non-streaming models add:
  - semantic_tokenizer encoder (for voice cloning feedback loop)
  - semantic_connector projection
  - lm_head for next-token prediction
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TraceableVAEDecoder(nn.Module):
    """Acoustic VAE decoder: latent (B, 1, vae_dim) -> audio (B, 1, samples)."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class TraceableVAEEncoder(nn.Module):
    """Acoustic VAE encoder: audio (B, 1, samples) -> mean (B, T, vae_dim).

    Returns only the mean (no sampling) for deterministic export.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, 1, samples) -> mean: (B, T, vae_dim)."""
        # The encoder returns an EncoderOutput with .mean and .std
        output = self.encoder(audio)
        return output.mean


class TraceableSemanticEncoder(nn.Module):
    """Semantic tokenizer encoder: audio (B, 1, samples) -> features (B, T, sem_dim)."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, 1, samples) -> features: (B, T, sem_dim)."""
        output = self.encoder(audio)
        return output.mean


class TraceableDiffusionHead(nn.Module):
    """Single DDPM denoising step."""

    def __init__(self, prediction_head: nn.Module) -> None:
        super().__init__()
        self.prediction_head = prediction_head

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.prediction_head(noisy_latent, timestep, condition=condition)


class TraceableLMHead(nn.Module):
    """LM head: hidden_state -> next-token logits."""

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (B, 1, hidden_size) -> logits: (B, 1, vocab_size)."""
        return self.lm_head(hidden_state)


class TraceableAcousticConnector(nn.Module):
    """Projects acoustic latent to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.connector(latent)


class TraceableSemanticConnector(nn.Module):
    """Projects semantic features to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.connector(features)
