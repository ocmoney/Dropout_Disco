# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/regression/data_utils.py
# Description: Utilities for Regression data prep (feature generation, dataset).
# Created: 2025-04-15
# Updated: 2025-04-15

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Add pandas if reading data here

def average_title_embedding(title_tokens, embeddings, vocab, embedding_dim):
    """Averages embeddings for tokens in a title."""
    
    # TODO: Align tokenization with how word2vec vocab was built
    # Example assumes simple split and vocab lookup
    
    vectors = []
    for token in title_tokens:
        idx = vocab.get(token, 0) # Get index, default to UNK (index 0)
        # TODO: Get vector based on how embeddings are stored (e.g., dict, numpy array, torch tensor)
        # Example: If embeddings is a numpy array keyed by index
        # vector = embeddings[idx] 
        # vectors.append(vector)
        pass # Placeholder

    if not vectors:
        # Return zero vector or average UNK vector if no known words
        return np.zeros(embedding_dim) 
        
    # Placeholder logic - replace with actual vector retrieval and averaging
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Fallback if vector retrieval failed but tokens existed
        return np.zeros(embedding_dim) 


# TODO: Define PyTorch Dataset for HN data
class HNDataset(Dataset):
    def __init__(self, features, targets):
        # features: Tensor of shape [num_samples, num_features]
        # targets: Tensor of shape [num_samples]
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

