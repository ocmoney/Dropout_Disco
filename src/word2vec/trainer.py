# Hacker News Upvote Prediction
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# Description: Handles Word2Vec training using Negative Sampling.
# Created: 2024-04-15
# Updated: 2024-04-16 # Updated date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F # For loss calculation
import os
from tqdm import tqdm
from typing import List, Optional, Union # Keep Union for type hint
from utils import logger
from .model import CBOW, SkipGram # Import both models
from .vocabulary import Vocabulary # Import Vocabulary for sampling

# W&B Import Handling
try:
    import wandb
    # Optional: Define WandbRun type for hinting if wandb is installed
    # from wandb.sdk.wandb_run import Run as WandbRun
except ImportError:
    logger.info("wandb not installed, W&B logging disabled in trainer.")
    wandb = None
    # WandbRun = None # Define as None if using type hints


def train_epoch_neg_sampling( # Renamed for clarity, or just replace train_epoch
    model: Union[CBOW, SkipGram], # Use Union type hint
    dataloader: DataLoader,
    criterion: nn.Module, # Expecting nn.BCEWithLogitsLoss
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    vocab: Vocabulary, # Need vocabulary for sampling
    k: int, # Number of negative samples
    model_type: str # 'CBOW' or 'SkipGram'
) -> float:
    """
    Trains the model for one epoch using Negative Sampling loss.

    Args:
        model: The CBOW or SkipGram model instance.
        dataloader: Provides batches of (input_indices, positive_target_indices).
                    For CBOW: (context_indices, center_index)
                    For SkipGram: (center_index, context_index)
        criterion: The loss function (expecting BCEWithLogitsLoss).
        optimizer: The optimizer.
        device: The device to train on.
        epoch_num: Current epoch number.
        total_epochs: Total number of epochs.
        vocab: The Vocabulary object for drawing negative samples.
        k (int): Number of negative samples per positive example.
        model_type (str): Specifies 'CBOW' or 'SkipGram'.

    Returns:
        The average loss per positive example for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning(f"Epoch {epoch_num+1}: Dataloader is empty.")
        return 0.0

    data_iterator = tqdm(
        dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs}",
        leave=False, unit="batch"
    )

    for batch_idx, batch_data in enumerate(data_iterator):
        optimizer.zero_grad()

        # 1. Get Input Embeddings & Positive Indices
        if model_type == "CBOW":
            context_indices, pos_indices = batch_data # Target is positive center word
            context_indices, pos_indices = context_indices.to(device), pos_indices.to(device)
            in_vectors = model.forward(context_indices) # CBOW forward gives avg context emb
        elif model_type == "SkipGram":
            center_indices, pos_indices = batch_data # Context is positive target
            center_indices, pos_indices = center_indices.to(device), pos_indices.to(device)
            in_vectors = model.forward(center_indices) # SkipGram forward gives center emb
        else:
             logger.error(f"Unsupported model_type: {model_type}"); continue

        # 2. Get Negative Sample Indices
        # --- FIX: Pass pos_indices directly if it's already (batch_size,) ---
        # The get_negative_samples function expects (batch_size,) or needs update
        if pos_indices.dim() == 1:
             neg_indices = vocab.get_negative_samples(pos_indices, k).to(device)
        # --- If CBOW target could be multi-dim (e.g. predicting multiple words), handle here ---
        # elif model_type == "CBOW" and pos_indices.dim() > 1:
        #     # Example: flatten or handle specific exclusion needed
        #     # logger.warning("Handling multi-dim pos_indices for CBOW NS - using first")
        #     neg_indices = vocab.get_negative_samples(pos_indices[:, 0], k).to(device) # Example logic
        else:
             # Fallback or error if shape is unexpected
             logger.error(f"Unexpected shape for pos_indices: {pos_indices.shape}")
             continue # Skip batch

        if neg_indices.numel() == 0 and k > 0: logger.warning(f"Batch {batch_idx}: Failed neg samples."); continue

        # 3. Get Output Embeddings for Positive and Negative Targets
        # Shape: (batch_size, embed_dim) for positive
        pos_out_vectors = model.forward_output_embeds(pos_indices)
        # Shape: (batch_size, k, embed_dim) for negative
        neg_out_vectors = model.forward_output_embeds(neg_indices)

        # 4. Calculate Scores (Dot Products)
        # Positive score: (batch_size,)
        pos_scores = torch.sum(in_vectors * pos_out_vectors, dim=1)
        # Negative scores: (batch_size, k)
        # Reshape in_vectors: (batch_size, 1, embed_dim) for broadcasting
        neg_scores = torch.sum(in_vectors.unsqueeze(1) * neg_out_vectors, dim=2)

        # 5. Calculate Binary Cross-Entropy Loss
        # Target for positive examples is 1
        pos_targets = torch.ones_like(pos_scores)
        # Target for negative examples is 0
        neg_targets = torch.zeros_like(neg_scores)

        pos_loss = criterion(pos_scores, pos_targets)
        neg_loss = criterion(neg_scores, neg_targets)

        # Combine losses
        loss = pos_loss + neg_loss

        # 6. Backpropagate and Optimize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        if batch_idx % 50 == 0: # Log less often
            # Average loss per positive example in batch
            avg_batch_loss_per_pos = batch_loss / (1 + k)
            data_iterator.set_postfix(loss=f"{avg_batch_loss_per_pos:.4f}")

    # Return average loss per positive example for the epoch
    average_loss = total_loss / (num_batches * (1 + k)) if k > 0 and num_batches > 0 else total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


def train_model(
    model: Union[CBOW, SkipGram], # Use Union type hint
    dataloader: DataLoader,
    criterion: nn.Module, # Should be BCEWithLogitsLoss for NS
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_save_dir: str,
    vocab: Vocabulary, # Pass vocab
    k: int, # Pass num negative samples
    model_type: str, # Pass model type
    wandb_run = None # Type hint: Optional[WandbRun] = None
) -> List[float]:
    """
    Orchestrates model training using Negative Sampling.
    Logs metrics to W&B, saves model state, returns epoch losses.

    Args:
        model: The CBOW or SkipGram model instance.
        dataloader: DataLoader for training data.
        criterion: The loss function (expecting BCEWithLogitsLoss).
        optimizer: The optimizer.
        device: The device to train on.
        epochs: Number of epochs to train.
        model_save_dir (str): Directory to save the model state.
        vocab (Vocabulary): Vocabulary object for negative sampling.
        k (int): Number of negative samples per positive example.
        model_type (str): 'CBOW' or 'SkipGram'.
        wandb_run: Optional W&B run object.

    Returns:
        List[float]: List of average epoch losses.
    """
    logger.info(
        f"🚀 Starting {model_type} training with Negative Sampling (k={k}): "
        f"{epochs} epochs on {device.type.upper()}"
    )
    model.to(device)
    epoch_losses = []

    if wandb_run and wandb: # Check if wandb was imported successfully
    
        try:
            wandb.watch(model, log="all", log_freq=100) # Log gradients every 100 batches
            logger.info("📊 W&B watching model gradients and parameters.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initiate wandb.watch: {e}")

    for epoch in range(epochs):
        # Call the negative sampling epoch trainer
        avg_epoch_loss = train_epoch_neg_sampling(
            model, dataloader, criterion, optimizer, device, epoch, epochs,
            vocab, k, model_type # Pass extra args
        )
        logger.info(
            f"✅ Epoch {epoch+1}/{epochs} | Avg Loss (NS): {avg_epoch_loss:.4f}"
        )
        epoch_losses.append(avg_epoch_loss)

        # Log metrics to W&B
        if wandb_run:
            try:
                log_data = {"epoch": epoch + 1, "avg_loss": avg_epoch_loss}
                current_lr = optimizer.param_groups[0]['lr']
                log_data["learning_rate"] = current_lr
                wandb_run.log(log_data)
                logger.debug(f"  Logged metrics to W&B for epoch {epoch+1}.")
            except Exception as e:
                logger.error(f"❌ W&B log failed epoch {epoch+1}: {e}")

    logger.info("🏁 Training finished.")
    # Save model state locally
    try:
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_file = os.path.join(model_save_dir, "model_state.pth")
        torch.save(model.state_dict(), model_save_file)
        logger.info(f"💾 Model state saved locally to: {model_save_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save model state locally: {e}")

    return epoch_losses