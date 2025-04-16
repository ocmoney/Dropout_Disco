# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/__init__.py
# Description: Initializes the word2vec package.
# Created: 2025-04-15
# Updated: 2025-04-15

# Expose key components for easier importing if desired
from .model import CBOW, SkipGram
from .vocabulary import Vocabulary
from .dataset import create_cbow_pairs, CBOWDataset
from .dataset import create_skipgram_pairs, SkipGramDataset
from .trainer import train_model

__all__ = ['CBOW', 'Vocabulary', 'create_cbow_pairs', 'CBOWDataset', 'train_model', 
              'train_epoch', 'SkipGram', 'create_skipgram_pairs', 'SkipGramDataset']

# You can add a logger statement here if needed, but it's often kept minimal.
from utils import logger
logger.debug("Word2vec package initialized.")