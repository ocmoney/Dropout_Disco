# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/vocabulary.py
# Description: Handles vocabulary creation, mapping, and persistence.
# Created: 2025-04-15
# Updated: 2025-04-16

import json
import os
import torch
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
# Assumes utils is importable from the project root or PYTHONPATH is set
from utils import logger

class Vocabulary:
    """Manages the mapping between words and numerical indices."""

    def __init__(self, unk_token: str = "<UNK>", min_freq: int = 5):
        """
        Initializes the Vocabulary.

        Args:
            unk_token (str): Token for unknown words. Defaults to "<UNK>".
            min_freq (int): Min frequency to include word. Defaults to 5.
        """
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.word_freq: Counter = Counter()
        self.unk_token = unk_token
        self.unk_index = -1 # Set during build_vocab
        self.min_freq = min_freq
        self.sampling_weights: Optional[torch.Tensor] = None
        logger.debug(
            f"⚒️ Vocabulary initialized (min_freq={min_freq}, "
            f"unk='{unk_token}')"
        )

    def build_vocab(self, words: List[str], ns_exponent: float = 0.75):
        """
        Builds vocab & calculates negative sampling weights.

        Args:
            words (List[str]): Words in the corpus.
            ns_exponent (float): Exponent for unigram distribution smoothing.
        """

        # --- Build Vocabulary ---
        logger.info(f"Building vocabulary from {len(words):,} words...")
        self.word_freq = Counter(words)
        logger.info(f"Found {len(self.word_freq):,} unique raw words.")

        # Init with UNK token
        self.idx2word = [self.unk_token]
        self.word2idx = {self.unk_token: 0}
        self.unk_index = 0

        # Add words meeting frequency threshold
        words_added = 0
    
        for word, freq in self.word_freq.items(): # Original order
            if freq >= self.min_freq and word != self.unk_token:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
                words_added += 1

        logger.info(
            f"📚 Vocabulary built: {len(self):,} entries "
            f"(incl. {self.unk_token}) with min_freq={self.min_freq}."
        )
        logger.debug(f"  {words_added} words added (excluding UNK).")
        logger.debug(
            f"  Top 10 frequent words: {self.word_freq.most_common(10)}"
        )

        # --- Calculate Negative Sampling Weights ---
        logger.info(
           f"Calculating negative sampling weights (exponent={ns_exponent})..."
        )
        pow_freqs = []
        # Use idx2word order to ensure weights align with indices
        for word in self.idx2word:
            freq = self.word_freq.get(word, 0) # Get freq from original counter
            # Skip UNK token for negative sampling? Often done.
            # Or include it with low probability if desired. Let's skip it.
            if word == self.unk_token:
                 pow_freqs.append(0.0) # Assign 0 weight to UNK
            else:
                 pow_freqs.append(math.pow(freq, ns_exponent))

        if sum(pow_freqs) == 0:
             logger.error("❌ Sum of powered frequencies is zero! ",
                          "Cannot create sampling distribution.")
             self.sampling_weights = None
             return # Exit if weights are unusable

        # Create tensor and normalize (implicitly handled by torch.multinomial)
        self.sampling_weights = torch.tensor(pow_freqs, dtype=torch.float)
        logger.info("Sampling weights calculated.")
        # Note: torch.multinomial uses these weights directly. 
        # No explicit normalization needed here.

    def get_index(self, word: str) -> int:
        """Returns index of a word, or UNK index if not found."""
        return self.word2idx.get(word, self.unk_index)

    def get_word(self, index: int) -> str:
        """Returns word for a given index, or UNK token if invalid."""
        if 0 <= index < len(self.idx2word):
            return self.idx2word[index]
        logger.warning(f"Index {index} out of bounds for vocabulary.")
        return self.unk_token

    def __len__(self) -> int:
        """Returns vocabulary size (number of unique words incl. UNK)."""
        return len(self.idx2word)

    def save_vocab(self, file_path: str):
        """Saves vocabulary mappings (word2idx, idx2word) to JSON."""
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'unk_token': self.unk_token,
            'min_freq': self.min_freq,
            'unk_index': self.unk_index,
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Vocabulary saved successfully to: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save vocab: {e}", exc_info=True)

    @classmethod
    def load_vocab(cls, file_path: str) -> 'Vocabulary':
        """
        Loads vocabulary mappings from a JSON file.

        Note: This currently does *not* load word frequencies or
        pre-calculated negative sampling weights. These need to be
        recalculated from the original corpus if required after loading.

        Args:
            file_path (str): Path to the vocabulary JSON file.

        Returns:
            Vocabulary: An instance populated with loaded data.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            Exception: For other potential loading/parsing errors.
        """
        logger.info(f"Attempting to load vocabulary from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Create a new instance using loaded or default parameters
            # Important: Ensure constructor defaults match saving defaults
            vocab = cls(
                unk_token=vocab_data.get('unk_token', "<UNK>"),
                min_freq=vocab_data.get('min_freq', 5) # Load saved min_freq
            )

            # Populate the core mappings
            vocab.word2idx = vocab_data.get('word2idx', {})
            vocab.idx2word = vocab_data.get('idx2word', [])
            # Ensure unk_index is correctly set, default to 0 if missing
            vocab.unk_index = vocab_data.get(
                'unk_index',
                 vocab.word2idx.get(vocab.unk_token, 0) # Fallback lookup
            )

            # Check basic integrity
            if not vocab.idx2word or vocab.idx2word[vocab.unk_index] != vocab.unk_token:
                 logger.error(f"❌ Loaded vocabulary integrity check failed", 
                              "(UNK token mismatch or empty).")
                 raise ValueError("Loaded vocabulary data is inconsistent.")
            if len(vocab.idx2word) != len(vocab.word2idx):
                 logger.error(f"❌ Loaded vocabulary integrity check failed", 
                              "(idx/word map size mismatch).")
                 raise ValueError("Loaded vocabulary data is inconsistent.")


            # --- Handle Missing Sampling Weights ---
            # Set sampling_weights to None explicitly, as they weren't saved
            vocab.sampling_weights = None
            logger.warning(
                "Loaded vocabulary does not contain sampling weights. "
                "Need to call build_vocab() again on the corpus "
                "if negative sampling is required."
            )

            logger.info(f"📚 Vocab loaded ({len(vocab):,} words) from {file_path}")
            return vocab

        except FileNotFoundError:
            logger.error(f"❌ Vocabulary file not found: {file_path}")
            raise # Re-raise the specific error
        except KeyError as e:
            logger.error(f"❌ Missing key '{e}' in vocabulary file: {file_path}")
            raise ValueError(f"Missing key '{e}' in vocabulary file.") from e
        except Exception as e:
            logger.error(f"❌ Failed to load/parse vocab from {file_path}: {e}",
                         exc_info=True)
            raise # Re-raise other exceptions

    def get_negative_samples(
            self,
            # Positive indices are primarily for context/logging now, not exclusion
            positive_indices: torch.Tensor,
            num_samples: int
        ) -> torch.Tensor:
            """
            Draws negative samples using the precomputed distribution.
            (Optimized version: Samples all at once, skips collision check)

            Args:
                positive_indices (torch.Tensor): Not used for exclusion here,
                                                shape (batch_size, [num_pos]).
                num_samples (int): Number of negative samples (k) to draw
                    per row in positive_indices.

            Returns:
                torch.Tensor: Indices of negative samples.
                    Shape: (batch_size, num_samples). Returns empty tensor if
                    sampling weights are missing.
            """
            if self.sampling_weights is None:
                logger.error("❌ Sampling weights missing. Cannot get samples.")
                return torch.empty((positive_indices.shape[0], 0),
                                dtype=torch.long,
                                device=positive_indices.device)

            if num_samples <= 0:
                return torch.empty((positive_indices.shape[0], 0),
                                dtype=torch.long,
                                device=positive_indices.device)

            batch_size = positive_indices.shape[0]
            total_negatives_needed = batch_size * num_samples

            # Sample all negatives at once
            # Assume sampling_weights are on CPU, move result to target device
            neg_indices_flat = torch.multinomial(
                self.sampling_weights, # Sampling weights tensor
                num_samples=total_negatives_needed,
                replacement=True # Ok to sample same neg word multiple times
            )

            # Reshape and move to the correct device
            neg_indices = neg_indices_flat.view(batch_size, num_samples).to(
                positive_indices.device # Ensure result is on same device
            )

            # Collision check removed for performance
            # Collisions are generally rare for large vocabs & moderate k

            return neg_indices