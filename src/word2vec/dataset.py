# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/dataset.py
# Description: Creates context-target pairs and PyTorch Dataset for CBOW.
# Created: 2024-04-15
# Updated: 2024-04-15

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import os
from utils import logger # Import project logger
from .vocabulary import Vocabulary # Import Vocabulary from the same package

def create_cbow_pairs(words: List[str], vocab: Vocabulary, window_size: int) -> List[Tuple[List[int], int]]:
    """
    Generates indexed context-target pairs for CBOW training.

    Args:
        words (List[str]): The sequence of words from the corpus.
        vocab (Vocabulary): The vocabulary object mapping words to indices.
        window_size (int): Number of context words on each side of the target word.

    Returns:
        List[Tuple[List[int], int]]: A list where each element is a tuple
                                      containing (list_of_context_indices, target_index).
                                      Words not in vocab are mapped to UNK index.
    """
    data_pairs = []
    # Convert words to indices once, handling unknown words
    indexed_words = [vocab.get_index(word) for word in words]
    logger.info(f"Converted {len(words):,} words to {len(indexed_words):,} indices.")

    # Iterate through words to create context-target pairs
    for i in range(window_size, len(indexed_words) - window_size):
        context_indices = indexed_words[i - window_size:i] + indexed_words[i + 1:i + window_size + 1]
        target_index = indexed_words[i]
        # Ensure context has the correct size (2 * window_size)
        if len(context_indices) == 2 * window_size:
            data_pairs.append((context_indices, target_index))
        # else: # Should not happen with the loop range, but good for debugging
        #     logger.warning(f"Skipping malformed context at index {i}")

    logger.info(f"Generated {len(data_pairs):,} context-target pairs with window size {window_size}.")
    if data_pairs:
         logger.debug(f"Sample pair (indices): Context={data_pairs[0][0]}, Target={data_pairs[0][1]}")
    return data_pairs


class CBOWDataset(Dataset):
    """PyTorch Dataset for CBOW training data (expects indexed pairs)."""

    def __init__(self, indexed_pairs: List[Tuple[List[int], int]]):
        """
        Initializes the dataset.

        Args:
            indexed_pairs (List[Tuple[List[int], int]]): Pre-generated list of
                                                        (context_indices, target_index) pairs.
        """
        self.indexed_pairs = indexed_pairs
        logger.debug(f"CBOWDataset initialized with {len(self.indexed_pairs)} pairs.")

    def __len__(self) -> int:
        """Returns the total number of context-target pairs."""
        return len(self.indexed_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a context-target pair by index.

        Args:
            idx (int): The index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - context_indices: Tensor of context word indices (dtype=torch.long).
                - target_index: Tensor of the target word index (dtype=torch.long).
        """
        context_indices, target_index = self.indexed_pairs[idx]
        # Convert lists/ints to tensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_index, dtype=torch.long)
        return context_tensor, target_tensor