# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: utils/__init__.py
# Description: Package initialization for utils module.
# Created: 2025-04-15
# Updated: 2025-04-15

# Import the configured logger instance from logging.py
# The setup_logging() function in logging.py runs automatically on this import.
from .logging import logger

# Import the get_device function from device_setup.py
from .device_setup import get_device

# Define what gets imported with 'from utils import *'
__all__ = ['logger', 'get_device']

# Optional: Log that the package is being initialized
# Note: logger might already be configured here due to import above
logger.debug("Utils package initialized (__init__.py).")