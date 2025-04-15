# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: train_word2vec.py
# Description: Main script to train the CBOW word2vec model on text8.
# Accepts hyperparameters via command-line arguments.
# Created: 2025-04-15
# Updated: 2025-04-15

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import json
import matplotlib.pyplot as plt


from utils import logger, get_device
from src.word2vec.model import CBOW
from src.word2vec.vocabulary import Vocabulary
from src.word2vec.dataset import create_cbow_pairs, CBOWDataset
from src.word2vec.trainer import train_model

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found at {config_path}.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading config file: {e}", exc_info=True)
        return None

def parse_arguments(config): # Ensure 'config' parameter is present
    """Parses command-line arguments, using loaded config for defaults."""
    parser = argparse.ArgumentParser(
        description="Train CBOW word2vec model.",
        # Allow overriding config values via command line
        conflict_handler='resolve'
    )
    # Get sections from config, providing empty dicts as fallback
    paths = config.get('paths', {})
    w2v_params = config.get('word2vec', {})
    train_params = config.get('training', {})

    # --- Argument Definitions using config values as defaults ---
    parser.add_argument(
        "--corpus-file", type=str,
        # Use config value, fallback to a hardcoded default if needed
        default=paths.get('corpus_file', "data/text8.txt"),
        help="Path to the input text corpus file."
    )
    # NOTE: Vocab file path is now dynamically generated in main()
    # We don't need an arg for the specific vocab file, just the base dir
    # parser.add_argument(
    #     "--vocab-file", type=str, default=paths.get('vocab_file'),
    #     help="Path to save/load the vocabulary JSON file."
    # )
    parser.add_argument(
        "--model-save-dir", type=str,
        default=paths.get('model_save_dir', "models/word2vec"),
        help="Base directory to save model artifacts (run-specific subdirs created)."
    )
    parser.add_argument(
        "--embed-dim", type=int,
        default=w2v_params.get('embedding_dim', 128), # Default from config or 128
        help="Dimensionality of word embeddings."
    )
    parser.add_argument(
        "--window-size", type=int,
        default=w2v_params.get('window_size', 3), # Default from config or 3
        help="Context window size (words on each side)."
    )
    parser.add_argument(
        "--min-freq", type=int,
        default=w2v_params.get('min_word_freq', 5), # Default from config or 5
        help="Minimum word frequency for vocabulary."
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=train_params.get('batch_size', 512), # Default from config or 512
        help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int,
        default=train_params.get('epochs', 10), # Default from config or 10
        help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float,
        default=train_params.get('learning_rate', 0.01), # Default from config or 0.01
        help="Learning rate for the Adam optimizer."
    )
    parser.add_argument(
        "--num-words", type=int,
        default=train_params.get('num_words_to_process', -1), # Default from config or -1 (all)
        help="Max number of words to process from corpus (-1 for all)."
    )
    parser.add_argument(
        "--force-rebuild-vocab", action='store_true', default=False, # Default is False
        help="Force rebuilding the vocabulary even if vocab file exists."
    )

    args = parser.parse_args()

    # Log the final effective configuration being used
    logger.info("--- Effective Configuration ---")
    for arg, value in vars(args).items():
         # Format arg name to match command line style
         arg_cli = "--" + arg.replace('_', '-')
         logger.info(f"  {arg_cli:<25}: {value}") # Align output
    logger.info("-----------------------------")

    return args

def save_losses(losses: list, save_dir: str):
    """Saves the list of epoch losses to a JSON file."""
    loss_file = os.path.join(save_dir, "cbow_training_losses.json")
    try:
        os.makedirs(save_dir, exist_ok=True) # Ensure dir exists
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump({'epoch_losses': losses}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {loss_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save training losses: {e}", exc_info=True)

def plot_losses(losses: list, save_dir: str):
    """Plots epoch losses and saves the plot."""
    if not losses:
        logger.warning("No losses recorded, skipping plot generation.")
        return
    plot_file = os.path.join(save_dir, "cbow_training_loss.png")
    try:
        os.makedirs(save_dir, exist_ok=True) # Ensure dir exists
        epochs = range(1, len(losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', linestyle='-')
        plt.title('CBOW Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.xticks(epochs)
        plt.grid(True, ls='--', linewidth=0.5)
        plt.savefig(plot_file)
        logger.info(f"📈 Training loss plot saved to: {plot_file}")
        plt.close() # Close plot to free memory
    except Exception as e:
        logger.error(f"❌ Failed to plot/save training losses: {e}", exc_info=True)

def format_num_words(num_words):
    """Formats large numbers for filenames (e.g., 10M, 500k, All)."""
    if num_words == -1:
        return "All"
    elif num_words >= 1_000_000:
        return f"{num_words // 1_000_000}M"
    elif num_words >= 1_000:
        return f"{num_words // 1_000}k"
    else:
        return str(num_words)

def main():
    """Main function to load config, parse args, and run training."""
    config = load_config()
    if config is None:
        logger.critical("🚨 Config file could not be loaded. Exiting.")
        return

    args = parse_arguments(config) # Get effective arguments

    logger.info("🚀 Starting Word2Vec CBOW Training Process...")

    # --- Setup Device ---
    device = get_device()

    # --- Define Dynamic Paths ---
    base_model_dir = args.model_save_dir
    corpus_name = os.path.splitext(os.path.basename(args.corpus_file))[0] # e.g., "text8"
    nw_str = format_num_words(args.num_words)

    # Vocabulary Filename (depends on corpus subset and min freq)
    vocab_filename = (
        f"{corpus_name}_vocab_NW{nw_str}_MF{args.min_freq}.json"
    )
    vocab_file_path = os.path.join(base_model_dir, vocab_filename)
    logger.info(f"Target Vocabulary File: {vocab_file_path}")

    # Run-Specific Subdirectory Name (based on key model/training params)
    run_name = (
        f"CBOW_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_"
        f"MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
    )
    run_save_dir = os.path.join(base_model_dir, run_name)
    logger.info(f"Run artifacts will be saved in: {run_save_dir}")

    # --- Load Corpus ---
    logger.info(f"📖 Loading corpus from: {args.corpus_file}")
    try:
        # (Corpus loading logic remains the same)
        with open(args.corpus_file, 'r', encoding='utf-8') as f:
            words = f.read().strip().split()
        logger.info(f"  Loaded {len(words):,} total words.")
        if args.num_words > 0 and len(words) > args.num_words:
             words = words[:args.num_words]
             logger.info(f"  Using first {len(words):,} words ({nw_str}).")
    except FileNotFoundError:
        logger.error(f"❌ Corpus file not found: {args.corpus_file}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading corpus: {e}", exc_info=True)
        return

    # --- Build or Load Vocabulary ---
    vocab = None
    if os.path.exists(vocab_file_path) and not args.force_rebuild_vocab: # Use constructed path
        try:
            vocab = Vocabulary.load_vocab(vocab_file_path) # Use constructed path
            if vocab.min_freq != args.min_freq:
                logger.warning(
                   f"Loaded vocab min_freq ({vocab.min_freq}) differs "
                   f"from requested ({args.min_freq}). Using loaded vocab."
                )
        except Exception:
            logger.error("Failed loading existing vocab, rebuilding...")
            vocab = None

    if vocab is None:
        logger.info(f"Building/Rebuilding vocabulary (min_freq={args.min_freq})...")
        vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build_vocab(words)
        vocab.save_vocab(vocab_file_path) # Use constructed path

    vocab_size = len(vocab)
    if vocab_size <= 1:
         logger.error("❌ Vocab only contains UNK. Check corpus/min_freq.")
         return
    logger.info(f"Vocabulary size: {vocab_size}")

    # --- Create Dataset & DataLoader ---
    # (Logic remains the same, uses args.window_size, args.batch_size etc.)
    logger.info("Generating context-target pairs...")
    indexed_pairs = create_cbow_pairs(words, vocab, args.window_size)
    if not indexed_pairs:
         logger.error("❌ No pairs generated. Check corpus/window size.")
         return

    logger.info("Creating PyTorch Dataset and DataLoader...")
    dataset = CBOWDataset(indexed_pairs)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type != 'mps')
    )
    logger.info(f"DataLoader created with batch size {args.batch_size}.")

    # --- Initialize Model, Loss, Optimizer ---
    # (Logic remains the same, uses args.embed_dim, args.lr etc.)
    logger.info("Initializing CBOW model...")
    model = CBOW(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model, Criterion, Optimizer (Adam, lr={args.lr}) ready.")

    # --- Train Model ---
    # Pass the run-specific directory for saving artifacts
    epoch_losses = train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_save_dir=run_save_dir # <-- Pass run-specific directory
    )

    # --- Save and Plot Losses ---
    # Save losses and plot inside the run-specific directory
    if epoch_losses:
        save_losses(epoch_losses, run_save_dir) # <-- Pass run-specific directory
        plot_losses(epoch_losses, run_save_dir) # <-- Pass run-specific directory
    else:
         logger.warning("Training did not return any losses.")

    logger.info(f"✅ Training run {run_name} completed.")
    logger.info(f"   Final model state: {os.path.join(run_save_dir, 'cbow_model_state.pth')}")
    logger.info(f"   Training losses: {os.path.join(run_save_dir, 'cbow_training_losses.json')}")
    logger.info(f"   Loss plot: {os.path.join(run_save_dir, 'cbow_training_loss.png')}")


if __name__ == "__main__":
    main()