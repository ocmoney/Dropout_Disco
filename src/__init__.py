# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/__init__.py
# Description: Initializes the word2vec package.
# Created: 2024-04-15
# Updated: 2024-04-15


# Makes the 'word2vec' directory a package.

# You can optionally import key classes/functions here for easier access
from .model import CBOW
from .vocabulary import Vocabulary
from .dataset import create_cbow_dataset
from .trainer import train_model

__all__ = ['CBOW', 'Vocabulary', 'create_cbow_dataset', 'train_model']