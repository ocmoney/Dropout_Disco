# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/__init__.py
# Description: Initializes the src package.
# Created: 2025-04-15
# Updated: 2025-04-15

try:
    from utils import logger
    logger.debug("SRC package initialized.")
except ImportError:
    # Handle case where utils might not be importable during initial scan
    pass

print("--- Running src/__init__.py ---") # Add this temporarily to confirm it runs