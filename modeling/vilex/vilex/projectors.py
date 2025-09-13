import random
from typing import List, Tuple

import torch
from torch import nn


class LinearProjector(nn.Module):
    """
    Simple linear projection layer for feature transformation.

    Projects input features from one dimension to another using a single linear layer.
    Commonly used as a baseline projector in vision-language models.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear projection to input features.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)

        Returns:
            Projected tensor of shape (batch_size, seq_len, out_dim)
        """
        return self.proj(x)


class AttentionPoolingProjector(nn.Module):
    """
    Attention-based pooling projector for vision features.

    Uses learnable query tokens to pool variable-length sequences of vision features
    into a fixed number of output tokens. Supports TailDrop regularization during training.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        num_heads: Number of attention heads (default: 8)
        num_output_tokens: Number of output tokens to produce (default: 1)
        taildrop_prob: Probability of applying TailDrop during training (default: 0.0)
        taildrop_max: Maximum number of tokens to drop with TailDrop (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_output_tokens: int = 1,
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.query = nn.Parameter(torch.randn(1, num_output_tokens, in_dim))
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x: torch.Tensor,k) -> torch.Tensor:
        """
        Apply attention pooling to input features.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)

        Returns:
            Pooled and projected tensor of shape (batch_size, num_output_tokens, out_dim)
            During training with TailDrop, may return fewer tokens.
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn(query, x, x)
        attn_out = attn_out[:, :-k, :]

        return self.proj(attn_out)


class MultiLayerAttentionPoolingProjector(nn.Module):
    """
    Multi-layer attention pooling projector for vision transformers with packed sequences.

    Extracts and pools features from multiple layers of a Vision Transformer backbone
    using attention pooling. Each selected layer's tokens are first projected by
    layer-specific linear layers before concatenation and final attention pooling.

    This projector works with packed sequence format where all tokens are concatenated
    into a single sequence dimension.

    Args:
        layer_indices: List of layer indices to extract features from
        in_dim: Input feature dimension from each layer
        out_dim: Output feature dimension
        num_layers: Number of layers (unused, kept for compatibility)
        num_heads: Number of attention heads for pooling (default: 8)
        num_output_tokens: Number of output tokens to produce (default: 1)
        taildrop_prob: Probability of applying TailDrop during training (default: 0.0)
        taildrop_max: Maximum number of tokens to drop with TailDrop (default: 0)
    """

    use_hidden_states = True  # Signal to VILEXModel to pass all hidden states

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layer_indices: List[int] = [1,2,3],
        num_layers: int = 4,
        num_heads: int = 8,
        num_output_tokens: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_output_tokens = num_output_tokens
        self.layer_indices = layer_indices

        # Layer-specific projectors: one for each selected layer
        self.layer_projectors = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in self.layer_indices])
        self.query = nn.Parameter(torch.randn(1, num_output_tokens, in_dim))
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, k,hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Pool features from multiple transformer layers using attention.

        Args:
            hidden_states: Tuple of layer outputs from ViT, each of shape (T, C)
                         where T=total tokens (packed format), C=embedding dimension

        Returns:
            Pooled tensor of shape (num_output_tokens, out_dim) in packed format
            During training with TailDrop, may return fewer tokens.
        """

        # Select and project features from specified layers
        layer_tokens = [proj(hidden_states[i]) for proj, i in zip(self.layer_projectors, self.layer_indices)]

        
        # Concatenate tokens from all selected layers along token dimension
        x = torch.cat(layer_tokens, dim=0)  # (T_total, C) where T_total = T * len(layer_indices)

        # Add batch dimension for MultiheadAttention
        x = x.unsqueeze(0)  # (1, T_total, C)

        # Use pre-defined query (already has batch dimension)
        query = self.query  # (1, num_output_tokens, in_dim)


        attn_out, _ = self.attn(query, x, x)  # (1, num_output_tokens, in_dim)

        # Apply TailDrop regularization during training
        if k > 0 and k < self.num_output_tokens:
            attn_out = attn_out[:, :-k, :]
        
        # Project to output dimension
        output = self.proj(attn_out)  # (1, num_output_tokens, out_dim)

        
        # Remove batch dimension to return packed format
        return output.squeeze(0)  # (num_output_tokens, out_dim)

