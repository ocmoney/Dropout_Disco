# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/model.py
# Description: PyTorch nn.Module definition for Word2Vec (CBOW/SkipGram).
# Created: 2025-04-15
# Updated: 2025-04-15

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Embedding for center words
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Embedding for context words (often treated differently)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Init weights (optional but can help)
        self.center_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)


    def forward(self, center_words, context_words, negative_words):
        """
        center_words: Tensor of shape (batch_size,)
        context_words: Tensor of shape (batch_size,)
        negative_words: Tensor of shape (batch_size, num_negative_samples)
        """
        # Get embeddings
        center_embeds = self.center_embeddings(center_words) # (batch_size, embed_dim)
        context_embeds = self.context_embeddings(context_words) # (batch_size, embed_dim)
        negative_embeds = self.context_embeddings(negative_words) # (batch_size, num_negative_samples, embed_dim)

        # Positive score (batch_size, 1)
        # Use batch matrix multiplication (bmm) for dot product
        pos_score = torch.bmm(center_embeds.unsqueeze(1), context_embeds.unsqueeze(2)).squeeze(2) # (batch_size, 1)

        # Negative score (batch_size, num_negative_samples)
        # Use batch matrix multiplication
        neg_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2) # (batch_size, num_negative_samples)

        # Calculate loss using log-sigmoid
        pos_loss = F.logsigmoid(pos_score).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1) # Sum over negative samples

        # Total loss (average over batch)
        total_loss = -(pos_loss + neg_loss).mean()
        return total_loss

    def get_embeddings(self):
        """Helper to get the primary (center) embeddings."""
        return self.center_embeddings.weight.data.cpu().numpy()


# TODO: Implement CBOWNegativeSampling if needed
class CBOWNegativeSampling(nn.Module):
     def __init__(self, vocab_size, embed_dim):
         super().__init__()
         # TODO: Implement CBOW architecture
         # Usually involves averaging context embeddings -> linear layer? Or just predict center?
         # Check original paper or common implementations
         pass

     def forward(self, context_words, center_word, negative_words):
         # TODO: Implement CBOW forward pass and loss
         pass

     def get_embeddings(self):
         # TODO: Return appropriate embeddings
         pass

