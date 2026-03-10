"""Traceable wrappers for VibeVoice-Realtime-0.5B TTS components.

Imports shared wrappers from common and adds model-specific ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from traceable_common import (  # noqa: E402, F401
    TraceableVAEDecoder,
    TraceableStreamingVAEDecoder,
    TraceableDiffusionHead,
    TraceableAcousticConnector,
)


class TraceableEOSClassifier(nn.Module):
    """Binary EOS classifier: hidden_state -> sigmoid probability."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (B, hidden_size) -> eos_prob: (B, 1)."""
        return torch.sigmoid(self.classifier(hidden_state))


class TraceableLMPrefill(nn.Module):
    """Base LM prefill: process a chunk of token embeddings, return hidden states."""

    def __init__(self, language_model: nn.Module, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """input_ids: (B, S) -> hidden_states: (B, S, hidden_size)."""
        inputs_embeds = self.embed_tokens(input_ids)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TraceableLMStep(nn.Module):
    """Single autoregressive LM step with explicit KV cache I/O."""

    def __init__(
        self,
        language_model: nn.Module,
        embed_tokens: nn.Module,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = embed_tokens
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache_position: torch.LongTensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = []
        for i in range(self.num_layers):
            past_key_values.append((key_cache[i], value_cache[i]))

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        new_keys = torch.stack([kv[0] for kv in outputs.past_key_values])
        new_values = torch.stack([kv[1] for kv in outputs.past_key_values])

        return outputs.last_hidden_state, new_keys, new_values
