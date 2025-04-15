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

# Assuming execution from project root (Dropout_Disco/)
from utils import logger, get_device
from src.word2vec.model import CBOW
from src.word2vec.vocabulary import Vocabulary
from src.word2vec.dataset import create_cbow_pairs, CBOWDataset
from src.word2vec.trainer import train_model

def parse_arguments():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train CBOW word2vec model."
    )
    parser.add_argument(
        "--corpus-file", type=str,
        default=os.environ.get("CORPUS_FILE", "data/text8.txt"),
        help="Path to the input text corpus file (e.g., text8)."
    )
    parser.add_argument(
        "--vocab-file", type=str,
        default=os.environ.get("VOCAB_FILE", "models/word2vec/text8_vocab.json"),
        help="Path to save/load the vocabulary JSON file."
    )
    parser.add_argument(
        "--model-save-dir", type=str,
        default=os.environ.get("MODEL_SAVE_DIR", "models/word2vec"),
        help="Directory to save the trained model state."
    )
    parser.add_argument(
        "--embed-dim", type=int,
        default=int(os.environ.get("EMBEDDING_DIM", 128)),
        help="Dimensionality of word embeddings."
    )
    parser.add_argument(
        "--window-size", type=int,
        default=int(os.environ.get("WINDOW_SIZE", 3)),
        help="Context window size (words on each side)."
    )
    parser.add_argument(
        "--min-freq", type=int,
        default=int(os.environ.get("MIN_WORD_FREQ", 5)),
        help="Minimum word frequency to include in vocabulary."
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=int(os.environ.get("BATCH_SIZE", 512)),
        help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int,
        default=int(os.environ.get("EPOCHS", 10)),
        help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float,
        default=float(os.environ.get("LEARNING_RATE", 0.01)),
        help="Learning rate for the Adam optimizer."
    )
    parser.add_argument(
        "--num-words", type=int,
        default=int(os.environ.get("NUM_WORDS_TO_PROCESS", 5_000_000)),
        help="Max number of words to process from corpus (-1 for all)."
    )
    parser.add_argument(
        "--force-rebuild-vocab", action='store_true',
        help="Force rebuilding the vocabulary even if vocab file exists."
    )

    args = parser.parse_args()
    return args

def main():
    """Main function to orchestrate the training process."""
    args = parse_arguments() # Get arguments from command line

    logger.info("🚀 Starting Word2Vec CBOW Training Process...")
    logger.info("--- Configuration ---")
    logger.info(f"  Corpus File:          {args.corpus_file}")
    logger.info(f"  Vocabulary File:      {args.vocab_file}")
    logger.info(f"  Model Save Directory: {args.model_save_dir}")
    logger.info(f"  Embedding Dimension:  {args.embed_dim}")
    logger.info(f"  Window Size:          {args.window_size}")
    logger.info(f"  Min Word Frequency:   {args.min_freq}")
    logger.info(f"  Batch Size:           {args.batch_size}")
    logger.info(f"  Epochs:               {args.epochs}")
    logger.info(f"  Learning Rate:        {args.lr}")
    logger.info(f"  Num Words to Process: {args.num_words if args.num_words > 0 else 'All'}")
    logger.info(f"  Force Rebuild Vocab:  {args.force_rebuild_vocab}")
    logger.info("-----------------------")


    # --- Setup Device ---
    device = get_device()

    # --- Load Corpus ---
    logger.info(f"📖 Loading corpus from: {args.corpus_file}")
    try:
        with open(args.corpus_file, 'r', encoding='utf-8') as f:
            words = f.read().strip().split()
        logger.info(f"  Loaded {len(words):,} total words.")
        if args.num_words > 0 and len(words) > args.num_words:
             words = words[:args.num_words]
             logger.info(f"  Using first {len(words):,} words for training.")
    except FileNotFoundError:
        logger.error(f"❌ Corpus file not found: {args.corpus_file}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading corpus: {e}", exc_info=True)
        return

    # --- Build or Load Vocabulary ---
    vocab = None
    if os.path.exists(args.vocab_file) and not args.force_rebuild_vocab:
        try:
            vocab = Vocabulary.load_vocab(args.vocab_file)
            # Optional: Check consistency
            if vocab.min_freq != args.min_freq:
                logger.warning(
                   f"Loaded vocab min_freq ({vocab.min_freq}) differs "
                   f"from requested ({args.min_freq}). Using loaded vocab."
                )
            # You could force rebuild here if params don't match desired config
        except Exception: # Broad except if loading fails badly
            logger.error("Failed to load existing vocab, rebuilding...")
            vocab = None # Ensure vocab is None to trigger rebuild

    if vocab is None:
        logger.info("Building new vocabulary...")
        vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build_vocab(words)
        vocab.save_vocab(args.vocab_file) # Save the built vocab

    vocab_size = len(vocab)
    if vocab_size <= 1: # Only UNK token
         logger.error("❌ Vocabulary only contains UNK token. Check corpus/min_freq.")
         return
    logger.info(f"Vocabulary size: {vocab_size}")

    # --- Create Dataset & DataLoader ---
    logger.info("Generating context-target pairs...")
    indexed_pairs = create_cbow_pairs(words, vocab, args.window_size)
    if not indexed_pairs:
         logger.error("❌ No context-target pairs. Check corpus/window size.")
         return

    logger.info("Creating PyTorch Dataset and DataLoader...")
    dataset = CBOWDataset(indexed_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0, # Recommended for MPS
        pin_memory=(device.type != 'mps') # Disable for MPS
    )
    logger.info(f"DataLoader created with batch size {args.batch_size}.")

    # --- Initialize Model, Loss, Optimizer ---
    logger.info("Initializing CBOW model...")
    model = CBOW(vocab_size=vocab_size, embedding_dim=args.embed_dim)
    criterion = nn.CrossEntropyLoss() # Standard loss for full softmax
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model, Criterion, Optimizer (Adam, lr={args.lr}) ready.")

    # --- Train Model ---
    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_save_dir=args.model_save_dir # Pass save dir
    )

    logger.info("✅ Word2Vec CBOW training process completed.")

if __name__ == "__main__":
    main()