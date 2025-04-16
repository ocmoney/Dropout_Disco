# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/dataset.py
# Description: Creates context-target pairs and PyTorch Dataset for CBOW.
# Created: 2025-04-15
# Updated: 2025-04-15 # Adjust date if modified

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
# Assumes utils is importable from the project root
from utils import logger
# Relative import for Vocabulary within the same package
from .vocabulary import Vocabulary

def create_cbow_pairs(
    words: List[str], vocab: Vocabulary, window_size: int
) -> List[Tuple[List[int], int]]:
    """
    Generates indexed context-target pairs for CBOW training.

    Maps words outside the vocabulary to the UNK index.

    Args:
        words (List[str]): Sequence of words from the corpus.
        vocab (Vocabulary): Vocabulary object mapping words to indices.
        window_size (int): Number of context words on each side.

    Returns:
        List[Tuple[List[int], int]]: List of (context_indices, target_index).
    """
    data_pairs = []
    # Convert words to indices once, handling unknown words
    indexed_words = [vocab.get_index(word) for word in words]
    logger.info(f"Converted {len(words):,} words to indices.")
    num_indices = len(indexed_words)

    # Iterate through words to create context-target pairs
    for i in range(window_size, num_indices - window_size):
        # Context is words before & after target i
        context_before = indexed_words[i - window_size : i]
        context_after = indexed_words[i + 1 : i + window_size + 1]
        context_indices = context_before + context_after
        target_index = indexed_words[i]

        # Ensure context has the correct size (e.g., near start/end)
        if len(context_indices) == 2 * window_size:
            data_pairs.append((context_indices, target_index))

    logger.info(f"Generated {len(data_pairs):,} context-target pairs "
                f"(window size={window_size}).")
    if data_pairs:
         logger.debug(
             f"Sample pair (indices): Context={data_pairs[0][0]}, "
             f"Target={data_pairs[0][1]}"
         )
    else:
        logger.warning("No context-target pairs were generated!")
    return data_pairs


class CBOWDataset(Dataset):
    """PyTorch Dataset for CBOW training data (expects indexed pairs)."""

    def __init__(self, indexed_pairs: List[Tuple[List[int], int]]):
        """
        Initializes the dataset with pre-generated indexed pairs.

        Args:
            indexed_pairs: List of (context_indices, target_index) tuples.
        """
        if not indexed_pairs:
            logger.warning("Initializing CBOWDataset with empty data!")
        self.indexed_pairs = indexed_pairs
        logger.debug(f"CBOWDataset created with {len(self.indexed_pairs)} items.")

    def __len__(self) -> int:
        """Returns the total number of context-target pairs."""
        return len(self.indexed_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a context-target pair as tensors by index.

        Args:
            idx (int): The index of the pair.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (context_indices, target_index)
        """
        context_indices, target_index = self.indexed_pairs[idx]
        # Convert lists/ints to tensors for PyTorch model
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_index, dtype=torch.long)
        return context_tensor, target_tensor
    

def create_skipgram_pairs(
    words: List[str], vocab: Vocabulary, window_size: int
) -> List[Tuple[int, int]]:
    """
    Generates indexed center-context pairs for Skip-gram training.

    Each pair consists of (center_word_index, context_word_index).

    Args:
        words (List[str]): Sequence of words from the corpus.
        vocab (Vocabulary): Vocabulary object mapping words to indices.
        window_size (int): Number of context words on each side.

    Returns:
        List[Tuple[int, int]]: List of (center_index, context_index) pairs.
    """
    data_pairs = []
    indexed_words = [vocab.get_index(word) for word in words]
    logger.info(f"Converted {len(words):,} words to indices for Skip-gram.")
    num_indices = len(indexed_words)

    for i in range(window_size, num_indices - window_size):
        center_word_index = indexed_words[i]
        # Generate pairs for each word in the context window
        for j in range(i - window_size, i + window_size + 1):
            if i == j: # Skip the center word itself
                continue
            context_word_index = indexed_words[j]
            data_pairs.append((center_word_index, context_word_index))

    logger.info(f"Generated {len(data_pairs):,} Skip-gram pairs "
                f"(window size={window_size}).")
    if data_pairs:
        logger.debug(
            f"Sample pair (indices): Center={data_pairs[0][0]}, "
            f"Context={data_pairs[0][1]}"
        )
    else:
        logger.warning("No Skip-gram pairs were generated!")
    return data_pairs

class SkipGramDataset(Dataset):
    """PyTorch Dataset for Skip-gram training data (center, context pairs)."""

    def __init__(self, indexed_pairs: List[Tuple[int, int]]):
        """
        Initializes the dataset with pre-generated indexed pairs.

        Args:
            indexed_pairs: List of (center_index, context_index) tuples.
        """
        if not indexed_pairs:
            logger.warning("SkipGramDataset initialized empty!")
        self.indexed_pairs = indexed_pairs
        logger.debug(f"SkipGramDataset created: {len(self.indexed_pairs)} items.")

    def __len__(self) -> int:
        """Returns the total number of center-context pairs."""
        return len(self.indexed_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a center-context pair as tensors by index.

        Args:
            idx (int): The index of the pair.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (center_index, context_index)
        """
        center_index, context_index = self.indexed_pairs[idx]
        center_tensor = torch.tensor(center_index, dtype=torch.long)
        context_tensor = torch.tensor(context_index, dtype=torch.long)
        
        # Input is center word, target is a context word
        return center_tensor, context_tensor