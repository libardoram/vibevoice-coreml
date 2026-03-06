"""Traceable wrappers for VibeVoice-ASR-HF components.

The ASR model has these convertible components:
  - acoustic_tokenizer encoder: audio -> latent features
  - semantic_tokenizer encoder: audio -> semantic features
  - acoustic_connector: latent -> LM embedding
  - semantic_connector: semantic -> LM embedding
  - lm_head: hidden_state -> text token logits

The Qwen2 LLM backbone (7B) is the largest component and
requires separate handling with KV cache / quantization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TraceableAcousticEncoder(nn.Module):
    """Acoustic tokenizer encoder: audio (B, 1, samples) -> mean (B, T, vae_dim).

    Processes in streaming mode with 60s segments for long audio.
    For CoreML export, we trace a fixed-length single-segment version.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.encoder = tokenizer.encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, 1, samples) -> mean: (B, T, vae_dim)."""
        output = self.encoder(audio)
        return output.mean


class TraceableSemanticEncoder(nn.Module):
    """Semantic tokenizer encoder: audio (B, 1, samples) -> features (B, T, sem_dim)."""

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.encoder = tokenizer.encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        output = self.encoder(audio)
        return output.mean


class TraceableAcousticConnector(nn.Module):
    """Projects acoustic latent to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, vae_dim) -> embedding: (B, T, hidden_size)."""
        return self.connector(features)


class TraceableSemanticConnector(nn.Module):
    """Projects semantic features to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.connector(features)


class TraceableLMHead(nn.Module):
    """LM head: hidden_state -> token logits."""

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_state)
