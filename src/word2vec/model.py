# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/model.py
# Description: Defines Word2Vec model architectures (CBOW and Skip-gram).
# Created: 2025-04-15
# Updated: 2025-04-16

import torch
import torch.nn as nn
from utils import logger


class CBOW(nn.Module):
    """
    CBOW model, structured for Negative Sampling.

    Learns input embeddings ('in_embed') for context words and output
    embeddings ('out_embed') used for predicting the center word during
    loss calculation (typically via dot products in NS loss).
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes input and output embedding layers.

        Args:
            vocab_size (int): The total number of unique words.
            embedding_dim (int): The desired dimensionality of embeddings.
        """
        super().__init__()
        # Input embeddings (lookup table for context words)
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (lookup table used for target prediction in NS)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Initialization (optional, PyTorch has defaults)
        # Consider initializing for potentially better/faster convergence
        init_range = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed.weight.data.uniform_(-init_range, init_range)
        logger.debug(f"CBOW model created: V={vocab_size}, D={embedding_dim}")

    def forward_context_embed(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates the averaged context embedding from input embeddings.
        This is the primary representation used for prediction in CBOW NS.

        Args:
            context_indices (torch.Tensor): Indices of context words.
                                            Shape: (batch_size, context_size).

        Returns:
            torch.Tensor: Averaged context embedding.
                          Shape: (batch_size, embedding_dim).
        """
        # Shape: (batch_size, context_size, embedding_dim)
        embedded = self.in_embed(context_indices)
        # Shape: (batch_size, embedding_dim)
        return embedded.mean(dim=1)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Primary forward pass for CBOW with Negative Sampling.

        Returns the averaged context embedding, which will be used externally
        (in the trainer) along with output embeddings to calculate NS loss.

        Args:
            context_indices (torch.Tensor): Indices of context words.
                                            Shape: (batch_size, context_size).

        Returns:
            torch.Tensor: Averaged context embedding.
                          Shape: (batch_size, embedding_dim).
        """
        return self.forward_context_embed(context_indices)

    def forward_output_embeds(self, indices: torch.Tensor) -> torch.Tensor:
         """Gets the output embeddings for given indices (center/negative)."""
         # Shape: (batch_size, [num_indices,] embedding_dim)
         return self.out_embed(indices)


class SkipGram(nn.Module):
    """
    Skip-gram model, structured for Negative Sampling.

    Learns input embeddings ('in_embed') for center words and output
    embeddings ('out_embed') used for predicting context words during
    loss calculation (typically via dot products in NS loss).
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes input (center) and output (context) embedding layers.

        Args:
            vocab_size (int): The total number of unique words.
            embedding_dim (int): The desired dimensionality of embeddings.
        """
        super().__init__()
        # Input embeddings (lookup table for center word)
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (lookup table used for context prediction in NS)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Initialization (optional)
        init_range = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        # Paper suggested zero init for output weights, but uniform works too
        self.out_embed.weight.data.uniform_(-init_range, init_range)
        # self.out_embed.weight.data.zero_() # Alternative init
        logger.debug(f"SkipGram model created: V={vocab_size}, D={embedding_dim}")


    def forward_input_embed(self, center_indices: torch.Tensor) -> torch.Tensor:
        """
        Gets the input embedding(s) for the center word(s).
        This serves as the primary representation for prediction in SkipGram NS.

        Args:
            center_indices (torch.Tensor): Indices of center words.
                                           Shape: (batch_size,).

        Returns:
            torch.Tensor: Center word embedding(s).
                          Shape: (batch_size, embedding_dim).
        """
        # Shape: (batch_size, embedding_dim)
        return self.in_embed(center_indices)

    def forward(self, center_indices: torch.Tensor) -> torch.Tensor:
        """
        Primary forward pass for SkipGram with Negative Sampling.

        Returns the center word embedding, which will be used externally
        (in the trainer) along with output embeddings to calculate NS loss.

        Args:
            center_indices (torch.Tensor): Indices of center words.
                                           Shape: (batch_size,).

        Returns:
            torch.Tensor: Center word embedding(s).
                          Shape: (batch_size, embedding_dim).
        """
        return self.forward_input_embed(center_indices)

    def forward_output_embeds(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Gets the output embeddings for given indices (context/negative).

        Args:
            indices (torch.Tensor): Indices of context words (positive or
                                    negative samples).
                                    Shape: (batch_size,) or (batch_size, k).

        Returns:
            torch.Tensor: Output embedding(s) for the specified indices.
                         Shape: (batch_size, embedding_dim) or
                                (batch_size, k, embedding_dim).
        """
        # Shape depends on input shape, e.g.,
        # (batch_size, embedding_dim) or (batch_size, k, embedding_dim)
        return self.out_embed(indices)