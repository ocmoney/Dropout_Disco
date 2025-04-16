# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: utils/run_utils.py
# Description: Utility functions for running experiments, saving artifacts.
# Created: 2025-04-16
# Updated: 2025-04-16

import os
import json
import yaml
import matplotlib.pyplot as plt
from typing import List # Import List for type hint

# Import logger from the logging module within the same package
from .logging import logger

def format_num_words(num_words: int) -> str:
    """Formats large numbers for filenames (e.g., 10M, 500k, All)."""
    if num_words == -1:
        return "All"
    elif num_words >= 1_000_000:
        return f"{num_words // 1_000_000}M"
    elif num_words >= 1_000:
        return f"{num_words // 1_000}k"
    else:
        return str(num_words)

def load_config(config_path: str = "config.yaml") -> dict | None:
    """Loads configuration from a YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found at {config_path}.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading config file: {e}", exc_info=True)
        return None

def save_losses(
    losses: List[float], save_dir: str, filename: str = "training_losses.json"
) -> str | None:
    """Saves the list of epoch losses to a JSON file."""
    if not os.path.isdir(save_dir):
         os.makedirs(save_dir, exist_ok=True)
    loss_file = os.path.join(save_dir, filename)
    try:
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump({'epoch_losses': losses}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {loss_file}")
        return loss_file # Return path for artifact logging
    except Exception as e:
        logger.error(f"❌ Failed to save training losses: {e}", exc_info=True)
        return None

def plot_losses(
    losses: List[float], save_dir: str, filename: str = "training_loss.png"
) -> str | None:
    """Plots epoch losses and saves the plot."""
    if not losses:
        logger.warning("No losses provided, skipping plot generation.")
        return None
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plot_file = os.path.join(save_dir, filename)
    try:
        epochs_range = range(1, len(losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, losses, marker='o', linestyle='-')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.xticks(epochs_range) # Ensure integer ticks for epochs
        plt.grid(True, ls='--', linewidth=0.5)
        plt.savefig(plot_file)
        logger.info(f"📈 Training loss plot saved to: {plot_file}")
        plt.close() # Close plot to free memory
        return plot_file # Return path for artifact logging
    except Exception as e:
        logger.error(f"❌ Failed to plot/save training losses: {e}", exc_info=True)
        return None