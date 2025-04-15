# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/trainer.py
# Description: Handles the training process for the word2vec model.
# Created: 2025-04-15
# Updated: 2025-04-15 # Adjust date if modified

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import List
from utils import logger
from .model import CBOW

def train_epoch(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int, # Renamed for clarity
    total_epochs: int # Added for progress bar display
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The CBOW model instance.
        dataloader: DataLoader providing context-target batches.
        criterion: The loss function (e.g., CrossEntropyLoss).
        optimizer: The optimizer (e.g., Adam).
        device: The device to train on (CPU or MPS).
        epoch_num: Current epoch number (starting from 0).
        total_epochs: Total number of epochs for progress display.

    Returns:
        The average loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning(f"Epoch {epoch_num+1}: Dataloader is empty, skipping epoch.")
        return 0.0

    # Progress bar setup
    data_iterator = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num+1}/{total_epochs}",
        leave=False, # Keeps bar on screen after loop if False
        unit="batch"
    )

    for batch_idx, (context, target) in enumerate(data_iterator):
        # Move data to the selected device
        context, target = context.to(device), target.to(device)

        # Standard training steps
        optimizer.zero_grad()
        output_logits = model(context)
        loss = criterion(output_logits, target)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        # Update progress bar description less frequently for performance
        if batch_idx % 20 == 0:
            data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

    average_loss = total_loss / num_batches
    return average_loss


def train_model(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_save_dir: str = "models/word2vec"
) -> List[float]:
    """
    Orchestrates model training and returns epoch losses.

    Args:
        # ... (same args as before) ...
        model_save_dir (str): Directory to save the trained model state.

    Returns:
        List[float]: A list containing the average loss for each epoch.
    """
    logger.info(
        f"🚀 Starting CBOW training: {epochs} epochs on {device.type.upper()}"
    )
    model.to(device)
    epoch_losses = []

    for epoch in range(epochs):
        avg_epoch_loss = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, epochs
        )
        logger.info(
            f"✅ Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f}"
        )
        epoch_losses.append(avg_epoch_loss)

    logger.info("🏁 Training finished.")

    try:
        os.makedirs(model_save_dir, exist_ok=True) # Ensure run directory exists
        # Use a standard name within the run directory
        model_save_file = os.path.join(model_save_dir, "model_state.pth")
        torch.save(model.state_dict(), model_save_file)
        logger.info(f"💾 Model state saved to: {model_save_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save model state: {e}", exc_info=True)

    return epoch_losses