# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/model.py
# Description: Defines the CBOW model architecture.
# Created: 2025-04-15
# Updated: 2025-04-16

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
    

class SkipGram(nn.Module):
    """
    Skip-gram model implementation (with full Softmax).

    Predicts context words based on a center word embedding.
    Note: This basic version uses a shared linear layer for all context
          predictions and full softmax via CrossEntropyLoss, which is
          computationally intensive for large vocabularies. Negative
          Sampling is the typical optimization not implemented here yet.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes the SkipGram model layers.

        Args:
            vocab_size (int): The total number of unique words.
            embedding_dim (int): The desired dimensionality of embeddings.
        """
        super().__init__()
        # Embedding layer for the center word (input)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # A single output layer to predict context words.
        # Its weights can be thought of as "output" or "context" embeddings.
        # Often, people tie weights (self.center_embeddings.weight) here,
        # but using a separate Linear layer is also common.
        self.output_linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word_indices: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for Skip-gram.

        Args:
            center_word_indices (torch.Tensor): Tensor of center word indices.
                                               Shape: (batch_size,).

        Returns:
            torch.Tensor: Logits over vocabulary for predicting context words.
                          Shape: (batch_size, vocab_size).
        """
        # center_embedded shape: (batch_size, embedding_dim)
        center_embedded = self.center_embeddings(center_word_indices)
        
        # output_logits shape: (batch_size, vocab_size)
        # Predict scores for context words based on the center embedding
        output_logits = self.output_linear(center_embedded)

        return output_logits