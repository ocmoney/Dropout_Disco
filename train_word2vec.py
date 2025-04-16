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
from torch.utils.data import Dataset, DataLoader
import argparse
import yaml
import json
import matplotlib.pyplot as plt
import wandb


from utils import logger, get_device
from src.word2vec.model import CBOW, SkipGram
from src.word2vec.vocabulary import Vocabulary
from src.word2vec.dataset import create_cbow_pairs, CBOWDataset
from src.word2vec.dataset import create_skipgram_pairs, SkipGramDataset
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

# train_word2vec.py
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# Description: Main script to train Word2Vec (CBOW or SkipGram) model.
# Accepts hyperparameters via command-line arguments and config file.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
import argparse
import yaml
import json
import matplotlib.pyplot as plt
import wandb

# --- Project-specific imports ---
from utils import logger, get_device
# Import both model classes
from src.word2vec.model import CBOW, SkipGram
from src.word2vec.vocabulary import Vocabulary
# Import both dataset classes and pair creation functions
from src.word2vec.dataset import (
    create_cbow_pairs, CBOWDataset,
    create_skipgram_pairs, SkipGramDataset
)
# Import the modified trainer
from src.word2vec.trainer import train_model

# --- Helper Functions ---
# (Keep format_num_words, load_config, save_losses, plot_losses as before)
def format_num_words(num_words):
    if num_words == -1: return "All"
    if num_words >= 1_000_000: return f"{num_words // 1_000_000}M"
    if num_words >= 1_000: return f"{num_words // 1_000}k"
    return str(num_words)

def load_config(config_path="config.yaml"):
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e: logger.error(f"❌ Error loading config: {e}"); return None

def save_losses(losses: list, save_dir: str, filename="training_losses.json"):
    loss_file = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump({'epoch_losses': losses}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {loss_file}")
        return loss_file
    except Exception as e: logger.error(f"❌ Failed to save losses: {e}"); return None

def plot_losses(losses: list, save_dir: str, filename="training_loss.png"):
    if not losses: return None
    plot_file = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(losses) + 1)
        plt.figure(figsize=(10, 6)); plt.plot(epochs, losses, marker='o', linestyle='-')
        plt.title('Training Loss per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Average Loss')
        plt.xticks(epochs); plt.grid(True, ls='--', linewidth=0.5); plt.savefig(plot_file)
        logger.info(f"📈 Training loss plot saved to: {plot_file}"); plt.close()
        return plot_file
    except Exception as e: logger.error(f"❌ Failed to plot losses: {e}"); return None


def parse_arguments(config):
    """Parses command-line arguments, using loaded config for defaults."""
    parser = argparse.ArgumentParser(description="Train Word2Vec model.", conflict_handler='resolve')
    paths = config.get('paths', {})
    w2v_params = config.get('word2vec', {})
    train_params = config.get('training', {})

    parser.add_argument("--corpus-file", type=str, default=paths.get('corpus_file', "data/text8.txt"), help="Path to corpus.")
    parser.add_argument("--model-save-dir", type=str, default=paths.get('model_save_dir', "models/word2vec"), help="Base save directory.")
    # --- Word2Vec Specific Args ---
    parser.add_argument(
        "--model-type", type=str,
        default=w2v_params.get('model_type', "CBOW"), # Default from config
        choices=['CBOW', 'SkipGram'], # Restrict choices
        help="Word2Vec model type (CBOW or SkipGram)."
    )
    parser.add_argument("--embed-dim", type=int, default=w2v_params.get('embedding_dim', 128), help="Embedding dimension.")
    parser.add_argument("--window-size", type=int, default=w2v_params.get('window_size', 3), help="Context window size.")
    parser.add_argument("--min-freq", type=int, default=w2v_params.get('min_word_freq', 5), help="Min word frequency.")
    # --- Training Args ---
    parser.add_argument("--batch-size", type=int, default=train_params.get('batch_size', 512), help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=train_params.get('epochs', 10), help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=train_params.get('learning_rate', 0.01), help="Learning rate.")
    parser.add_argument("--num-words", type=int, default=train_params.get('num_words_to_process', -1), help="Max words (-1 for all).")
    parser.add_argument("--force-rebuild-vocab", action='store_true', default=False, help="Force rebuilding vocabulary.")
    # --- W&B Args ---
    parser.add_argument('--wandb-project', type=str, default='dropout-disco-word2vec', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Custom W&B run name')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Disable W&B logging')

    args = parser.parse_args()
    logger.info("--- Effective Configuration ---")
    for arg, value in vars(args).items(): logger.info(f"  --{arg.replace('_', '-'):<25}: {value}")
    logger.info("-----------------------------")
    return args


def save_losses(losses: list, save_dir: str, filename="cbow_training_losses.json"):
    """Saves the list of epoch losses to a JSON file."""
    loss_file = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump({'epoch_losses': losses}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {loss_file}")
        return loss_file # Return path for artifact logging
    except Exception as e:
        logger.error(f"❌ Failed to save training losses: {e}", exc_info=True)
        return None

def plot_losses(losses: list, save_dir: str, filename="cbow_training_loss.png"):
    """Plots epoch losses and saves the plot."""
    if not losses: return None
    plot_file = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
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
        plt.close()
        return plot_file # Return path for artifact logging
    except Exception as e:
        logger.error(f"❌ Failed to plot/save training losses: {e}", exc_info=True)
        return None

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
    """Main function to load config, parse args, init W&B, and run training."""
    config = load_config()
    if config is None:
        logger.critical("🚨 Configuration file could not be loaded. Exiting.")
        return

    args = parse_arguments(config)

    # --- Initialize W&B ---
    run = None # Initialize run object
    if not args.no_wandb:
        try:
            # Construct a dynamic run name if not provided
            if not args.wandb_run_name:
                 nw_str = format_num_words(args.num_words)
                 run_name = f"CBOW_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
            else:
                 run_name = args.wandb_run_name

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity, # Your username/team, optional
                config=vars(args),       # Log all hyperparameters from args
                name=run_name,           # Set a descriptive name for the run
                save_code=True           # Optional: Save main script to W&B
            )
            logger.info(f"📊 Initialized W&B run: {run.name} (Project: {args.wandb_project})")
            logger.info(f"  View run online at: {run.get_url()}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize W&B: {e}. Proceeding without W&B logging.", exc_info=True)
            run = None # Ensure run is None if init fails
    else:
        logger.info("📊 W&B logging disabled via --no-wandb flag.")


    logger.info("🚀 Starting Word2Vec CBOW Training Process...")

    # --- Setup Device ---
    device = get_device()

    # --- Define Dynamic Paths ---
    # (Path definition logic remains the same)
    base_model_dir = args.model_save_dir
    corpus_name = os.path.splitext(os.path.basename(args.corpus_file))[0]
    nw_str = format_num_words(args.num_words)
    vocab_filename = f"{corpus_name}_vocab_NW{nw_str}_MF{args.min_freq}.json"
    vocab_file_path = os.path.join(base_model_dir, vocab_filename)
    run_name_fs = f"CBOW_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
    run_save_dir = os.path.join(base_model_dir, run_name_fs)
    model_save_file = os.path.join(run_save_dir, "model_state.pth") # Define model save path

    logger.info(f"Target Vocabulary File: {vocab_file_path}")
    logger.info(f"Run artifacts directory: {run_save_dir}")

    # --- Load Corpus, Build/Load Vocab, Create Dataset ---
    # (This logic remains mostly the same, using args...)
    # ... (paste relevant loading/vocab/dataset code here, ensure it uses args) ...
    logger.info(f"📖 Loading corpus from: {args.corpus_file}")
    try:
        with open(args.corpus_file, 'r', encoding='utf-8') as f: words = f.read().strip().split()
        logger.info(f"  Loaded {len(words):,} total words.")
        if args.num_words > 0 and len(words) > args.num_words:
             words = words[:args.num_words]
             logger.info(f"  Using first {len(words):,} words ({nw_str}).")
    except Exception as e: logger.error(f"❌ Error loading corpus: {e}"); return

    vocab = None
    if os.path.exists(vocab_file_path) and not args.force_rebuild_vocab:
        try: vocab = Vocabulary.load_vocab(vocab_file_path)
        except Exception: logger.error("Failed loading vocab, rebuilding..."); vocab = None
    if vocab is None:
        logger.info(f"Building/Rebuilding vocabulary (min_freq={args.min_freq})...")
        vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build_vocab(words); vocab.save_vocab(vocab_file_path)
    vocab_size = len(vocab)
    if vocab_size <= 1: logger.error("❌ Vocab only UNK."); return
    logger.info(f"Vocabulary size: {vocab_size}")

    # --- Create Dataset & DataLoader based on model_type ---
    logger.info(f"Generating {args.model_type} training pairs...")
    dataset: Dataset # Type hint for clarity
    if args.model_type == "CBOW":
        indexed_pairs = create_cbow_pairs(words, vocab, args.window_size)
        if not indexed_pairs: logger.error("❌ No CBOW pairs."); return
        dataset = CBOWDataset(indexed_pairs)
    elif args.model_type == "SkipGram":
        indexed_pairs = create_skipgram_pairs(words, vocab, args.window_size)
        if not indexed_pairs: logger.error("❌ No SkipGram pairs."); return
        dataset = SkipGramDataset(indexed_pairs)
    else:
        logger.error(f"❌ Unknown model_type: {args.model_type}"); return

    logger.info("Creating PyTorch DataLoader...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type != 'mps'))
    logger.info(f"DataLoader ready with {len(dataset):,} pairs, batch size {args.batch_size}.")

    # --- Initialize Model based on model_type ---
    logger.info(f"Initializing {args.model_type} model...")
    model: nn.Module # Type hint
    if args.model_type == "CBOW":
        model = CBOW(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    elif args.model_type == "SkipGram":
        model = SkipGram(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    # Add else block if more models were possible

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model, Criterion, Optimizer (Adam, lr={args.lr}) ready.")

    # --- Train Model ---
    # NOTE: We need to modify train_model to accept the wandb run object for logging
    # Let's pass it for now, assuming trainer.py will be updated (or log here)
    epoch_losses = train_model( # train_model now defined below for simplicity
        model=model, dataloader=dataloader, criterion=criterion,
        optimizer=optimizer, device=device, epochs=args.epochs,
        model_save_dir=run_save_dir, wandb_run=run # Pass run object
    )

    # --- Save and Plot Losses (Locally) ---
    loss_file_path = None
    plot_file_path = None
    if epoch_losses:
        loss_file_path = save_losses(epoch_losses, run_save_dir)
        plot_file_path = plot_losses(epoch_losses, run_save_dir)
    else:
         logger.warning("Training did not return any losses.")

    # --- Log Artifacts to W&B ---
    if run and config is not None: # Check if W&B run was initialized
        logger.info("☁️ Logging artifacts to W&B...")
        try:
            # Log Model Checkpoint
            model_artifact = wandb.Artifact(f"cbow_model_{run.id}", type="model", description=f"CBOW model state trained for {args.epochs} epochs on text8 ({nw_str} words).")
            model_artifact.add_file(model_save_file)
            run.log_artifact(model_artifact)
            logger.info("  Logged model state artifact.")

            # Log Vocabulary
            vocab_artifact = wandb.Artifact(f"vocab_{run.id}", type="vocabulary", description=f"Vocabulary for text8 ({nw_str} words, min_freq={args.min_freq}).")
            vocab_artifact.add_file(vocab_file_path)
            run.log_artifact(vocab_artifact)
            logger.info("  Logged vocabulary artifact.")

            # Log Loss files (optional but good)
            if loss_file_path:
                loss_artifact = wandb.Artifact(f"losses_{run.id}", type="results")
                loss_artifact.add_file(loss_file_path)
                if plot_file_path:
                     loss_artifact.add_file(plot_file_path)
                run.log_artifact(loss_artifact)
                logger.info("  Logged losses.json and loss plot artifact.")

        except Exception as e:
            logger.error(f"❌ Failed to log artifacts to W&B: {e}", exc_info=True)

    # --- Finish W&B Run ---
    if run:
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ Word2Vec CBOW training process completed.")


if __name__ == "__main__":
    main()