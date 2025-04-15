# Hacker News Upvote Prediction
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/model.py
# Description: Defines the CBOW model architecture.
# Created: 2024-04-15
# Updated: 2024-04-15

import torch
import torch.nn as nn

class CBOW(nn.Module):
    """
    Continuous Bag-of-Words (CBOW) model implementation.

    Predicts a target word based on the average of its context word embeddings.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes the CBOW model layers.

        Args:
            vocab_size (int): The total number of unique words in the vocabulary.
            embedding_dim (int): The desired dimensionality of the word embeddings.
        """
        super().__init__()
        # Embedding layer: maps word indices to dense vectors
        # padding_idx=0 could be useful if we reserve index 0 for padding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer: maps averaged context embedding to vocabulary scores
        self.linear = nn.Linear(embedding_dim, vocab_size)
        # logger.debug(f"CBOW model initialized: vocab={vocab_size}, embed_dim={embedding_dim}") # Requires logger import if used here

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CBOW model.

        Args:
            context_indices (torch.Tensor): Tensor of context word indices (batch_size, context_size).

        Returns:
            torch.Tensor: Logits over the vocabulary for the target word (batch_size, vocab_size).
        """
        # Get embeddings for context words: (batch_size, context_size, embedding_dim)
        embedded = self.embeddings(context_indices)
        # Average context embeddings along the context_size dimension: (batch_size, embedding_dim)
        averaged_embedded = embedded.mean(dim=1)
        # Pass averaged embedding through the linear layer
        output_logits = self.linear(averaged_embedded)
        return output_logits