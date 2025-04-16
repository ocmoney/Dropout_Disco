# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/vocabulary.py
# Description: Handles vocabulary creation, mapping, and persistence.
# Created: 2025-04-15
# Updated: 2025-04-15 # Adjust date if modified

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
        positive_indices: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Draws negative samples, ensuring they don't match positive indices.

        Args:
            positive_indices (torch.Tensor): Indices of positive examples
                to exclude. Shape: (batch_size,) or (batch_size, num_pos).
                Expected on the same device sampling occurs (CPU default).
            num_samples (int): Number of negative samples (k) to draw
                per row in positive_indices.

        Returns:
            torch.Tensor: Indices of negative samples.
                Shape: (batch_size, num_samples). Returns empty tensor if
                sampling weights are missing or sampling fails.
        """
        if self.sampling_weights is None:
            logger.error("❌ Sampling weights missing. Cannot get samples.")
            # Return tensor with correct first dim, zero second dim
            return torch.empty((positive_indices.shape[0], 0),
                               dtype=torch.long,
                               device=positive_indices.device) # Match device

        if num_samples <= 0:
            return torch.empty((positive_indices.shape[0], 0),
                               dtype=torch.long,
                               device=positive_indices.device)

        batch_size = positive_indices.shape[0]
        # Flatten positive indices if it has multiple columns (e.g., CBOW target)
        # We want to avoid sampling ANY of the positive targets for a given item
        if positive_indices.dim() > 1:
            # This case might need more careful handling depending on exact shape
            # For now, assume positive_indices is (batch_size,) [SkipGram]
            # or requires specific exclusion logic if (batch_size, num_pos)
            logger.warning("Multi-dim positive_indices requires specific "
                           "exclusion logic not fully implemented here.")
            # Simple exclusion: Check against first positive index per row
            exclude_indices = positive_indices[:, 0]
        else:
            exclude_indices = positive_indices

        # Initialize output tensor for negative samples
        neg_candidates = torch.empty((batch_size, num_samples),
                                     dtype=torch.long,
                                     device=positive_indices.device)

        # Sample negatives iteratively for each item in the batch
        # to handle exclusions correctly. This is less vectorized but safer.
        for i in range(batch_size):
            positive_to_exclude = exclude_indices[i].item()
            tries = 0
            max_tries = num_samples * 5 # Safety limit for resampling

            # Sample k candidates for the i-th item
            item_negs = torch.multinomial(
                self.sampling_weights, # Assumed on CPU for now
                num_samples=num_samples,
                replacement=True
            ).to(positive_indices.device) # Move samples to target device

            # Check for collisions and resample if necessary
            collision_mask = (item_negs == positive_to_exclude)
            while collision_mask.any() and tries < max_tries:
                num_collisions = collision_mask.sum().item()
                # Resample only for the collided indices
                resamples = torch.multinomial(
                    self.sampling_weights,
                    num_samples=num_collisions,
                    replacement=True
                ).to(positive_indices.device)
                # Place resamples into the correct positions
                item_negs[collision_mask] = resamples
                # Check for new collisions
                collision_mask = (item_negs == positive_to_exclude)
                tries += num_collisions # Count total resample attempts

            if tries >= max_tries:
                logger.warning(f"Max resampling tries exceeded for item {i}."
                               " Some negative samples might match positive.")

            neg_candidates[i] = item_negs

        return neg_candidates