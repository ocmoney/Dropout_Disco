# Hacker News Upvote Prediction
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/vocabulary.py
# Description: Handles vocabulary creation, mapping, and persistence.
# Created: 2024-04-15
# Updated: 2024-04-15

import json
import os
from collections import Counter
from typing import List, Dict, Tuple
from utils import logger # Import project logger

class Vocabulary:
    """Manages the mapping between words and numerical indices."""

    def __init__(self, unk_token: str = "<UNK>", min_freq: int = 5):
        """
        Initializes the Vocabulary.

        Args:
            unk_token (str): The token to use for unknown words. Defaults to "<UNK>".
            min_freq (int): Minimum frequency for a word to be included in vocab. Defaults to 5.
        """
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.word_freq: Counter = Counter()
        self.unk_token = unk_token
        self.unk_index = -1 # Will be set during build
        self.min_freq = min_freq
        logger.debug(f"⚒️ Vocabulary initialized with min_freq={min_freq}, unk_token='{unk_token}'")

    def build_vocab(self, words: List[str]):
        """
        Builds the vocabulary from a list of words.

        Args:
            words (List[str]): A list of all words in the corpus.
        """
        logger.info(f"Building vocabulary from {len(words):,} words...")
        self.word_freq = Counter(words)
        logger.info(f"Found {len(self.word_freq):,} unique raw words.")

        # Initialize vocabulary with UNK token
        self.idx2word = [self.unk_token]
        self.word2idx = {self.unk_token: 0}
        self.unk_index = 0

        # Add words meeting frequency threshold
        words_added = 0
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                if word not in self.word2idx: # Should always be true after UNK init
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1
                    words_added += 1

        logger.info(f"📚 Vocabulary built: {len(self.idx2word):,} unique words (including {self.unk_token}) meeting min_freq={self.min_freq}.")
        logger.debug(f"  {words_added} words added (excluding UNK).")
        logger.debug(f"  Top 10 most frequent words: {self.word_freq.most_common(10)}")

    def get_index(self, word: str) -> int:
        """Returns the index of a word, or the UNK index if not found."""
        return self.word2idx.get(word, self.unk_index)

    def get_word(self, index: int) -> str:
        """Returns the word for a given index."""
        return self.idx2word[index] if 0 <= index < len(self.idx2word) else self.unk_token

    def __len__(self) -> int:
        """Returns the size of the vocabulary (number of unique words including UNK)."""
        return len(self.idx2word)

    def save_vocab(self, file_path: str):
        """Saves the vocabulary mappings to a JSON file."""
        dir_name = os.path.dirname(file_path)
        if dir_name:
             os.makedirs(dir_name, exist_ok=True)
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'unk_token': self.unk_token,
            'min_freq': self.min_freq,
            'unk_index': self.unk_index
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Vocabulary saved successfully to: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save vocabulary to {file_path}: {e}", exc_info=True)

    @classmethod
    def load_vocab(cls, file_path: str) -> 'Vocabulary':
        """Loads vocabulary mappings from a JSON file."""
        logger.info(f"Attempting to load vocabulary from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Create a new instance and populate it
            vocab = cls(unk_token=vocab_data.get('unk_token', "<UNK>"),
                        min_freq=vocab_data.get('min_freq', 5))
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
            vocab.unk_index = vocab_data.get('unk_index', vocab.word2idx.get(vocab.unk_token, 0))
            # Note: word_freq is not saved/loaded, rebuild if needed from corpus

            logger.info(f"📚 Vocabulary loaded successfully ({len(vocab):,} words).")
            return vocab
        except FileNotFoundError:
             logger.error(f"❌ Vocabulary file not found: {file_path}")
             raise
        except Exception as e:
             logger.error(f"❌ Failed to load vocabulary from {file_path}: {e}", exc_info=True)
             raise