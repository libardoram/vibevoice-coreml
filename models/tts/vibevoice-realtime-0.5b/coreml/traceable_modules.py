"""Traceable wrappers for VibeVoice TTS components.

Each wrapper isolates a component for torch.jit.trace, removing
dynamic control flow and returning only tensors.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TraceableVAEDecoder(nn.Module):
    """Acoustic VAE decoder: latent (B, 1, vae_dim) -> audio (B, 1, samples).

    Strips streaming cache logic — processes one frame at a time with
    explicit state passed in/out as tensors.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (B, 1, vae_dim) -> audio: (B, 1, frame_samples)."""
        return self.decoder(latent)


class TraceableDiffusionHead(nn.Module):
    """Single denoising step: (noisy, timestep, condition) -> predicted noise.

    The full diffusion loop (20 steps) is orchestrated externally.
    """

    def __init__(self, prediction_head: nn.Module) -> None:
        super().__init__()
        self.prediction_head = prediction_head

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_latent: (B, vae_dim)
            timestep: (B,) integer timesteps
            condition: (B, hidden_size) from TTS LM
        Returns:
            predicted noise/velocity: (B, vae_dim)
        """
        return self.prediction_head(noisy_latent, timestep, condition=condition)


class TraceableEOSClassifier(nn.Module):
    """Binary EOS classifier: hidden_state -> sigmoid probability."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (B, hidden_size) -> eos_prob: (B, 1)."""
        return torch.sigmoid(self.classifier(hidden_state))


class TraceableLMPrefill(nn.Module):
    """Base LM prefill: process a chunk of token embeddings, return hidden states.

    KV cache is handled internally by the Qwen2 model and returned as
    past_key_values. For CoreML we'll need to externalize this.
    """

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
    """Single autoregressive LM step with explicit KV cache I/O.

    For CoreML conversion, KV caches are passed as explicit tensor inputs/outputs
    rather than using the HuggingFace DynamicCache.
    """

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
        # KV caches passed as stacked tensors: (num_layers, B, num_kv_heads, max_seq, head_dim)
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states: (B, 1, hidden_size)
            new_key_cache: updated key cache
            new_value_cache: updated value cache
        """
        inputs_embeds = self.embed_tokens(input_ids)
        # Reconstruct past_key_values from flat tensors
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

        # Collect updated KV caches
        new_keys = torch.stack([kv[0] for kv in outputs.past_key_values])
        new_values = torch.stack([kv[1] for kv in outputs.past_key_values])

        return outputs.last_hidden_state, new_keys, new_values


class TraceableAcousticConnector(nn.Module):
    """Projects speech latent to LM embedding space."""

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, speech_latent: torch.Tensor) -> torch.Tensor:
        """speech_latent: (B, 1, vae_dim) -> embedding: (B, 1, hidden_size)."""
        return self.connector(speech_latent)
