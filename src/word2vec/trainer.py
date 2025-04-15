# Hacker News Upvote Prediction
# Copyright (c) 2024 Dropout Disco Team (Yurii, James, Ollie, Emil)
# Description: Handles the training process for the word2vec model, with W&B logging.
# Created: 2024-04-15
# Updated: 2024-04-15 # Adjust date if modified

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import List, Optional # Import Optional
from utils import logger
from .model import CBOW

try:
    import wandb
    from wandb.sdk.wandb_run import Run as WandbRun
except ImportError:
    logger.info("wandb not installed, W&B logging disabled in trainer.")
    wandb = None
    WandbRun = None


# --- train_epoch function remains the same ---
def train_epoch(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int
) -> float:
    """Trains the model for one epoch and returns the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning(f"Epoch {epoch_num+1}: Dataloader empty.")
        return 0.0

    data_iterator = tqdm(
        dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs}",
        leave=False, unit="batch"
    )

    for batch_idx, (context, target) in enumerate(data_iterator):
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output_logits = model(context)
        loss = criterion(output_logits, target)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss
        if batch_idx % 20 == 0:
            data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

    average_loss = total_loss / num_batches
    return average_loss


# --- train_model function MODIFIED ---
def train_model(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_save_dir: str,
    # Add wandb_run argument, make it optional
    wandb_run = None # Type hint: Optional[WandbRun] = None
) -> List[float]:
    """
    Orchestrates model training, logs metrics to W&B (if enabled),
    saves model state, and returns epoch losses.

    Args:
        model: The model to train.
        dataloader: The data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (CPU/MPS).
        epochs: Number of epochs to train.
        model_save_dir: Directory path for saving artifacts for this run.
        wandb_run: Optional W&B run object for logging metrics.

    Returns:
        List[float]: A list containing the average loss for each epoch.
    """
    logger.info(
        f"🚀 Starting CBOW training: {epochs} epochs on {device.type.upper()}"
    )
    model.to(device)
    epoch_losses = []

    # --- Watch model with W&B (optional) ---
    # This logs gradients and parameter distributions
    if wandb_run and wandb: # Check if wandb was imported successfully
        try:
            wandb.watch(model, log="all", log_freq=100) # Log gradients every 100 batches
            logger.info("📊 W&B watching model gradients and parameters.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initiate wandb.watch: {e}")

    for epoch in range(epochs):
        avg_epoch_loss = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, epochs
        )
        logger.info(
            f"✅ Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f}"
        )
        epoch_losses.append(avg_epoch_loss)

        # --- Log metrics to W&B ---
        if wandb_run: # Check if run object was passed and is valid
            try:
                # Log loss and potentially learning rate (if scheduler is used)
                log_data = {"epoch": epoch + 1, "avg_loss": avg_epoch_loss}
                # Example: Add learning rate if using a scheduler
                current_lr = optimizer.param_groups[0]['lr']
                log_data["learning_rate"] = current_lr
                wandb_run.log(log_data)
                logger.debug(f"  Logged metrics to W&B for epoch {epoch+1}.")
            except Exception as e:
                logger.error(
                   f"❌ Failed to log metrics to W&B for epoch {epoch+1}: {e}"
                )

    logger.info("🏁 Training finished.")
    # Save the trained model's state dictionary locally
    try:
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_file = os.path.join(model_save_dir, "model_state.pth")
        torch.save(model.state_dict(), model_save_file)
        logger.info(f"💾 Model state saved locally to: {model_save_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save model state locally: {e}")

    return epoch_losses