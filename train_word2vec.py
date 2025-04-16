# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: train_word2vec.py
# Description: Main script to train Word2Vec model using config and utils.
# Created: 2025-04-15
# Updated: 2025-04-16

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
# Removed yaml, json, plt imports as they are handled in utils now
import wandb

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

# --- Main Function (Imports helpers now) ---
def main():
    """Main function to orchestrate the training process."""
    # Load config using the utility function
    config = load_config() # Imported from utils
    if config is None: logger.critical("🚨 Config missing. Exiting."); return
    args = parse_arguments(config) # Keep local or move to utils? Keep here for now.

    # --- Initialize W&B (Remains the same) ---
    run = None
    if not args.no_wandb:
        try:
            nw_str = format_num_words(args.num_words) # Imported from utils
            if not args.wandb_run_name:
                 run_name = (
                     f"{args.model_type}_D{args.embed_dim}_W{args.window_size}_"
                     f"NW{nw_str}_MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
                 )
            else: run_name = args.wandb_run_name
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             config=vars(args), name=run_name, save_code=True)
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.get_url()})")
        except Exception as e: logger.error(f"❌ Failed W&B init: {e}"); run = None
    else: logger.info("📊 W&B logging disabled.")

    logger.info(f"🚀 Starting Word2Vec {args.model_type} Training...")
    device = get_device() # Imported from utils

    # --- Define Dynamic Paths (Uses format_num_words) ---
    base_model_dir = args.model_save_dir
    corpus_name = os.path.splitext(os.path.basename(args.corpus_file))[0]
    nw_str = format_num_words(args.num_words) # Imported from utils
    vocab_filename = f"{corpus_name}_vocab_NW{nw_str}_MF{args.min_freq}.json"
    vocab_file_path = os.path.join(base_model_dir, vocab_filename)
    run_name_fs = (
        f"{args.model_type}_D{args.embed_dim}_W{args.window_size}_NW{nw_str}_"
        f"MF{args.min_freq}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"
    )
    run_save_dir = os.path.join(base_model_dir, run_name_fs)
    model_save_file = os.path.join(run_save_dir, "model_state.pth")
    loss_json_filename = "training_losses.json" # Define filename for util func
    loss_plot_filename = "training_loss.png"   # Define filename for util func

    logger.info(f"Target Vocabulary File: {vocab_file_path}")
    logger.info(f"Run artifacts directory: {run_save_dir}")

    # --- Load Corpus (Remains the same) ---
    # ... (corpus loading logic) ...
    logger.info(f"📖 Loading corpus from: {args.corpus_file}")
    try:
        with open(args.corpus_file, 'r', encoding='utf-8') as f: words = f.read().strip().split()
        logger.info(f"  Loaded {len(words):,} total words.")
        if args.num_words > 0 and len(words) > args.num_words:
             words = words[:args.num_words]; logger.info(f"  Using first {len(words):,} words ({nw_str}).")
    except Exception as e: logger.error(f"❌ Error loading corpus: {e}"); return


    # --- Build or Load Vocabulary (Remains the same, uses vocab_file_path) ---
    # ... (vocab loading/building logic) ...
    vocab = None
    if os.path.exists(vocab_file_path) and not args.force_rebuild_vocab:
        try: vocab = Vocabulary.load_vocab(vocab_file_path)
        except Exception: logger.error("Failed loading vocab, rebuilding..."); vocab = None
    if vocab is None:
        logger.info(f"Building/Rebuilding vocab (min_freq={args.min_freq})..."); vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build_vocab(words); vocab.save_vocab(vocab_file_path)
    elif vocab.sampling_weights is None: # Check if weights needed rebuilding
         logger.warning("Rebuilding vocab to calculate sampling weights...")
         vocab.build_vocab(words); vocab.save_vocab(vocab_file_path)
    vocab_size = len(vocab)
    if vocab_size <= 1 or vocab.sampling_weights is None:
        logger.error("❌ Vocab invalid or missing sampling weights."); return
    logger.info(f"Vocabulary size: {vocab_size}")


    # --- Create Dataset & DataLoader (Remains the same) ---
    # ... (dataset/dataloader creation logic) ...
    logger.info(f"Generating {args.model_type} training pairs...")
    dataset: Dataset
    if args.model_type == "CBOW": indexed_pairs = create_cbow_pairs(words, vocab, args.window_size); dataset = CBOWDataset(indexed_pairs)
    elif args.model_type == "SkipGram": indexed_pairs = create_skipgram_pairs(words, vocab, args.window_size); dataset = SkipGramDataset(indexed_pairs)
    else: logger.error(f"❌ Unknown model_type: {args.model_type}"); return
    if not indexed_pairs: logger.error(f"❌ No {args.model_type} pairs generated."); return
    logger.info("Creating DataLoader..."); dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type != 'mps'))
    logger.info(f"DataLoader ready with {len(dataset):,} pairs, batch size {args.batch_size}.")


    # --- Initialize Model, Loss, Optimizer (Select BCE Loss) ---
    logger.info(f"Initializing {args.model_type} model...")
    model: nn.Module
    if args.model_type == "CBOW": model = CBOW(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    elif args.model_type == "SkipGram": model = SkipGram(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    criterion = nn.BCEWithLogitsLoss() # USE BCE LOSS FOR NEGATIVE SAMPLING
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model, Criterion (BCEWithLogitsLoss), Optimizer ready.")


    # --- Train Model (Passes k from args) ---
    epoch_losses = train_model(
        model=model, dataloader=dataloader, criterion=criterion,
        optimizer=optimizer, device=device, epochs=args.epochs,
        model_save_dir=run_save_dir,
        vocab=vocab, k=args.negative_samples, model_type=args.model_type,
        wandb_run=run
    )

    # --- Save/Plot Losses (Uses imported functions) ---
    loss_file_path = None; plot_file_path = None
    if epoch_losses:
        loss_file_path = save_losses(epoch_losses, run_save_dir, loss_json_filename) # Imported
        plot_file_path = plot_losses(epoch_losses, run_save_dir, loss_plot_filename) # Imported
    else: logger.warning("Training returned no losses.")

    # --- Log Artifacts to W&B (Remains the same) ---
    # ... (artifact logging logic) ...
    if run: # Only log artifacts if W&B run exists
        logger.info("☁️ Logging artifacts to W&B...")
        try:
            run_id = run.id if run else "local"
            model_artifact = wandb.Artifact(f"{args.model_type}_model_{run_id}", type="model"); model_artifact.add_file(model_save_file); run.log_artifact(model_artifact); logger.info("  Logged model artifact.")
            vocab_artifact = wandb.Artifact(f"vocab_{run_id}", type="vocabulary"); vocab_artifact.add_file(vocab_file_path); run.log_artifact(vocab_artifact); logger.info("  Logged vocabulary artifact.")
            if loss_file_path:
                results_artifact = wandb.Artifact(f"results_{run_id}", type="results"); results_artifact.add_file(loss_file_path)
                if plot_file_path: results_artifact.add_file(plot_file_path)
                run.log_artifact(results_artifact); logger.info("  Logged results artifact.")
        except Exception as e: logger.error(f"❌ Failed W&B artifact logging: {e}")


    # --- Finish W&B Run (Remains the same) ---
    if run: run.finish(); logger.info("☁️ W&B run finished.")

    logger.info(f"✅ {args.model_type} training process completed.")


if __name__ == "__main__":
    main()