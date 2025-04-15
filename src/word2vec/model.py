# src/word2vec/model.py
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# Description: Defines the CBOW model architecture.
# Created: 2024-04-15
# Updated: 2024-04-15 # Adjust date if modified

import torch
import torch.nn as nn
from utils import logger

class CBOW(nn.Module):
    """
    Continuous Bag-of-Words (CBOW) model implementation.

    Predicts a target word based on the average of its context word embeddings.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes the CBOW model layers.

        Args:
            vocab_size (int): The total number of unique words.
            embedding_dim (int): The desired dimensionality of embeddings.
        """
        super().__init__()
        # Input embeddings (lookup table)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output layer predicting the target word index
        self.linear = nn.Linear(embedding_dim, vocab_size)
        # Optional: Log initialization if logger passed or imported
        logger.debug(f"CBOW model created: V={vocab_size}, D={embedding_dim}")

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CBOW model.

        Args:
            context_indices (torch.Tensor): Tensor of context word indices
                                            Shape: (batch_size, context_size).

        Returns:
            torch.Tensor: Logits over vocabulary for the target word.
                          Shape: (batch_size, vocab_size).
        """
        # embedded shape: (batch_size, context_size, embedding_dim)
        embedded = self.embeddings(context_indices)
        # averaged_embedded shape: (batch_size, embedding_dim)
        averaged_embedded = embedded.mean(dim=1)
        # output_logits shape: (batch_size, vocab_size)
        output_logits = self.linear(averaged_embedded)
        # Optional: Log output shape if logger passed or imported
        logger.debug(
            f"Forward pass: context_indices={context_indices.shape}, "
            f"output_logits={output_logits.shape}"
        )

        return output_logits