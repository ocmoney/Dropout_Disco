# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/data_utils.py
# Description: Utilities for Word2Vec data prep (vocab, sampling, context/target).
# Created: 2025-04-15
# Updated: 2025-04-15

from collections import Counter
import random
import numpy as np
# Add other imports like torch.utils.data.Dataset if creating custom dataset

def build_vocab(tokens, min_freq=5, max_vocab_size=None):
    """Builds vocabulary from a list of tokens."""
    print(f"Building vocabulary...")
    word_counts = Counter(tokens)
    # Sort words by frequency, highest first
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

    # Apply max_vocab_size limit
    if max_vocab_size:
        sorted_words = sorted_words[:max_vocab_size-1] # Reserve one slot for UNK

    # Create word -> index mapping, include UNK token
    word_to_idx = {'<UNK>': 0}
    idx_to_word = {0: '<UNK>'}
    unk_count = 0

    for i, word in enumerate(sorted_words):
        if word_counts[word] >= min_freq:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        else:
            # Words below min_freq will be mapped to UNK
            unk_count += word_counts[word] # Count how many tokens become UNK

    print(f"  Vocabulary size: {len(word_to_idx)}")
    print(f"  Tokens mapped to <UNK>: Count represented by words below min_freq {min_freq}.")
    # Note: A more sophisticated count would track the actual number of UNK occurrences during tokenization

    # Optional: Create frequency list for negative sampling (unigram distribution ^ 3/4)
    # word_freqs = np.array([word_counts.get(idx_to_word.get(i, '<UNK>'), 0) for i in range(len(idx_to_word))])
    # unigram_dist = word_freqs**0.75
    # unigram_dist /= unigram_dist.sum()

    # return word_to_idx, idx_to_word, unigram_dist
    return word_to_idx, idx_to_word


def tokenize_text(file_path):
    """Reads and tokenizes text file."""
    print(f"Tokenizing text from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.split() # Simple space splitting
    print(f"  Found {len(tokens):,} tokens.")
    return tokens


def generate_skipgram_pairs(token_indices, window_size):
    """Generates (center_word_idx, context_word_idx) pairs for Skip-gram."""
    pairs = []
    n = len(token_indices)
    print(f"Generating Skip-gram pairs with window size {window_size}...")
    for i in range(n):
        center_word_idx = token_indices[i]
        # Determine actual window bounds, considering document edges
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        for j in range(start, end):
            if i == j: # Skip the center word itself
                continue
            context_word_idx = token_indices[j]
            pairs.append((center_word_idx, context_word_idx))
    print(f"  Generated {len(pairs):,} pairs.")
    return pairs

# TODO: Implement generate_cbow_pairs if needed
# TODO: Implement negative sampling logic (or use built-in if available)

