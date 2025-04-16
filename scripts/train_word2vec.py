# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: scripts/train_word2vec.py
# Description: Main script to train Word2Vec model using config and utils.
# Created: 2025-04-15
# Updated: 2025-04-16

import os
from dotenv import load_dotenv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import wandb


# Load environment variables from .env file
load_dotenv()

# --- Add project root to sys.path --- START ---
# This allows importing modules from 'src' and 'utils' when run from root
# Get the directory where this script resides (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root: Dropout_Disco/)
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
# --- Add project root to sys.path --- END ---

# --- Project-specific imports ---
# Import helpers from utils
from utils import (
    logger, get_device, load_config,
    save_losses, plot_losses, format_num_words
)
# Import core components
from src.word2vec.model import CBOW, SkipGram
from src.word2vec.vocabulary import Vocabulary
from src.word2vec.dataset import (
    create_cbow_pairs, CBOWDataset,
    create_skipgram_pairs, SkipGramDataset
)
from src.word2vec.trainer import train_model


# --- Argument Parsing (Remains Here) ---
def parse_arguments(config):
    """Parses command-line arguments, using loaded config for defaults."""
    # ... (Keep the full parse_arguments function definition here as before) ...
    # ... (Make sure it uses config properly) ...
    parser = argparse.ArgumentParser(description="Train Word2Vec model.", conflict_handler='resolve')
    paths = config.get('paths', {})
    w2v_params = config.get('word2vec', {})
    train_params = config.get('training', {})
    parser.add_argument("--corpus-file", type=str, default=paths.get('corpus_file', "data/text8.txt"), help="Path to corpus.")
    parser.add_argument("--model-save-dir", type=str, default=paths.get('model_save_dir', "models/word2vec"), help="Base save directory.")
    parser.add_argument("--model-type", type=str, default=w2v_params.get('model_type', "CBOW"), choices=['CBOW', 'SkipGram'], help="Word2Vec model type.")
    parser.add_argument("--embed-dim", type=int, default=w2v_params.get('embedding_dim', 128), help="Embedding dimension.")
    parser.add_argument("--window-size", type=int, default=w2v_params.get('window_size', 3), help="Context window size.")
    parser.add_argument("--min-freq", type=int, default=w2v_params.get('min_word_freq', 5), help="Min word frequency.")
    parser.add_argument("--negative-samples", type=int, default=w2v_params.get('negative_samples', 5), help="Number of negative samples (k).")
    parser.add_argument("--batch-size", type=int, default=train_params.get('batch_size', 512), help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=train_params.get('epochs', 10), help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=train_params.get('learning_rate', 0.01), help="Learning rate.")
    parser.add_argument("--num-words", type=int, default=train_params.get('num_words_to_process', -1), help="Max words (-1 for all).")
    parser.add_argument("--force-rebuild-vocab", action='store_true', default=False, help="Force rebuilding vocabulary.")
    parser.add_argument('--wandb-project', type=str, default='dropout-disco-word2vec', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Custom W&B run name')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Disable W&B logging')
    args = parser.parse_args()
    logger.info("--- Effective Configuration ---")
    for arg, value in vars(args).items(): logger.info(f"  --{arg.replace('_', '-'):<25}: {value}")
    logger.info("-----------------------------")
    return args

# --- Main Functions ---
def setup_experiment(args):
    """Initializes W&B, sets up device, defines paths."""
    logger.info("--- Setting up Experiment ---")
    device = get_device()
    run = None
    if not args.no_wandb:
        # ... (W&B init logic using args) ...
        try:
            nw_str = format_num_words(args.num_words)
            if not args.wandb_run_name:
                 run_name = f"{args.model_type}_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
            else: run_name = args.wandb_run_name
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name, save_code=True)
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.get_url()})")
        except Exception as e: logger.error(f"❌ Failed W&B init: {e}"); run = None
    else: logger.info("📊 W&B logging disabled.")

    # Define paths
    base_model_dir = args.model_save_dir
    corpus_name = os.path.splitext(os.path.basename(args.corpus_file))[0]
    nw_str = format_num_words(args.num_words)
    vocab_filename = f"{corpus_name}_vocab_NW{nw_str}_MF{args.min_freq}.json"
    vocab_file_path = os.path.join(base_model_dir, vocab_filename)
    run_name_fs = f"{args.model_type}_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
    run_save_dir = os.path.join(base_model_dir, run_name_fs)
    model_save_file = os.path.join(run_save_dir, "model_state.pth")
    loss_json_filename = "training_losses.json"
    loss_plot_filename = "training_loss.png"

    paths = {
        "vocab_file": vocab_file_path,
        "run_save_dir": run_save_dir,
        "model_save_file": model_save_file,
        "loss_json_file": os.path.join(run_save_dir, loss_json_filename),
        "loss_plot_file": os.path.join(run_save_dir, loss_plot_filename)
    }
    logger.info(f"Artifacts directory: {run_save_dir}")

    return device, run, paths


def prepare_data(args, paths):
    """Loads corpus, builds/loads vocab, creates dataset & dataloader."""
    logger.info("--- Preparing Data ---")
    # Load Corpus
    logger.info(f"📖 Loading corpus from: {args.corpus_file}")
    try:
        # ... (corpus loading logic using args.corpus_file, args.num_words) ...
        with open(args.corpus_file, 'r', encoding='utf-8') as f: words = f.read().strip().split()
        logger.info(f"  Loaded {len(words):,} total words.")
        nw_str = format_num_words(args.num_words) # Use helper
        if args.num_words > 0 and len(words) > args.num_words:
             words = words[:args.num_words]; logger.info(f"  Using first {len(words):,} words ({nw_str}).")
    except Exception as e: logger.error(f"❌ Error loading corpus: {e}"); return None, None, None

    # Build/Load Vocab
    vocab = None
    if os.path.exists(paths["vocab_file"]) and not args.force_rebuild_vocab:
        try: vocab = Vocabulary.load_vocab(paths["vocab_file"])
        except Exception: logger.error("Failed loading vocab, rebuilding..."); vocab = None
    if vocab is None:
        logger.info(f"Building/Rebuilding vocab (min_freq={args.min_freq})..."); vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build_vocab(words); vocab.save_vocab(paths["vocab_file"])
    elif vocab.sampling_weights is None: # Rebuild if loaded vocab lacks weights for NS
         logger.warning("Rebuilding vocab for sampling weights..."); vocab.build_vocab(words); vocab.save_vocab(paths["vocab_file"])
    vocab_size = len(vocab)
    if vocab_size <= 1 or vocab.sampling_weights is None: logger.error("❌ Vocab invalid/missing weights."); return None, None, None
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create Dataset & DataLoader
    logger.info(f"Generating {args.model_type} training pairs..."); dataset: Dataset
    if args.model_type == "CBOW": indexed_pairs = create_cbow_pairs(words, vocab, args.window_size); dataset = CBOWDataset(indexed_pairs)
    elif args.model_type == "SkipGram": indexed_pairs = create_skipgram_pairs(words, vocab, args.window_size); dataset = SkipGramDataset(indexed_pairs)
    else: logger.error(f"❌ Unknown model_type: {args.model_type}"); return None, None, None
    if not indexed_pairs: logger.error(f"❌ No {args.model_type} pairs generated."); return None, None, None

    logger.info("Creating DataLoader..."); dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(args.device.type != 'mps')) # Pass device from args
    logger.info(f"DataLoader ready with {len(dataset):,} pairs, batch size {args.batch_size}.")

    return vocab, dataloader, vocab_size


def initialize_model(args, vocab_size):
    """Initializes the model, criterion, and optimizer."""
    logger.info("--- Initializing Model ---")
    logger.info(f"Initializing {args.model_type} model (Embed Dim: {args.embed_dim})...")
    model: nn.Module
    if args.model_type == "CBOW": model = CBOW(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    elif args.model_type == "SkipGram": model = SkipGram(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    else: raise ValueError(f"Unknown model_type: {args.model_type}") # Should not happen if argparse choices used

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model, Criterion (BCEWithLogitsLoss), Optimizer (Adam, lr={args.lr}) ready.")
    return model, criterion, optimizer


def finalize_run(run, epoch_losses, paths, args, config):
    """Saves local artifacts, logs artifacts to W&B, finishes W&B run."""
    logger.info("--- Finalizing Run ---")
    loss_file_path = None; plot_file_path = None
    if epoch_losses:
        loss_file_path = save_losses(epoch_losses, paths["run_save_dir"]) # Use paths dict
        plot_file_path = plot_losses(epoch_losses, paths["run_save_dir"]) # Use paths dict
    else: logger.warning("Training returned no losses.")

    # Log Artifacts to W&B
    if run: # Check if W&B run exists
        logger.info("☁️ Logging artifacts to W&B...")
        try:
            # Add args/paths needed for logging descriptions
            nw_str = format_num_words(args.num_words)
            run_id = run.id
            model_save_file = paths["model_save_file"] # Get from paths
            vocab_file_path = paths["vocab_file"]     # Get from paths

            model_artifact = wandb.Artifact(f"{args.model_type}_model_{run_id}", type="model"); model_artifact.add_file(model_save_file); run.log_artifact(model_artifact); logger.info("  Logged model artifact.")
            vocab_artifact = wandb.Artifact(f"vocab_{run_id}", type="vocabulary"); vocab_artifact.add_file(vocab_file_path); run.log_artifact(vocab_artifact); logger.info("  Logged vocabulary artifact.")
            if loss_file_path:
                results_artifact = wandb.Artifact(f"results_{run_id}", type="results"); results_artifact.add_file(loss_file_path)
                if plot_file_path: results_artifact.add_file(plot_file_path)
                run.log_artifact(results_artifact); logger.info("  Logged results artifact.")
            # You could also log the config file itself as an artifact
            # config_artifact = wandb.Artifact("config"); config_artifact.add_file("config.yaml"); run.log_artifact(config_artifact)
        except Exception as e: logger.error(f"❌ Failed W&B artifact logging: {e}")

        # Finish W&B Run
        run.finish(); logger.info("☁️ W&B run finished.")

    logger.info(f"✅ {args.model_type} training process completed.")


def main():
    """Main function to orchestrate the training process."""
    config = load_config()
    if config is None: return
    args = parse_arguments(config)

    # Setup experiment (W&B, device, paths)
    device, run, paths = setup_experiment(args)
    args.device = device # Add device to args for convenience if needed later

    # Prepare data (load corpus, vocab, dataset, dataloader)
    vocab, dataloader, vocab_size = prepare_data(args, paths)
    if dataloader is None: return # Exit if data prep failed

    # Initialize model components
    model, criterion, optimizer = initialize_model(args, vocab_size)

    # Train the model
    epoch_losses = train_model(
        model=model, dataloader=dataloader, criterion=criterion,
        optimizer=optimizer, device=device, epochs=args.epochs,
        model_save_dir=paths["run_save_dir"], # Pass specific save dir
        vocab=vocab, k=args.negative_samples, model_type=args.model_type,
        wandb_run=run # Pass run object
    )

    # Finalize (save local artifacts, log W&B artifacts, finish run)
    finalize_run(run, epoch_losses, paths, args, config)


if __name__ == "__main__":
    main()