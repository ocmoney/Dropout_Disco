# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/trainer.py
# Description: Handles the training process for the word2vec model.
# Created: 2024-04-15
# Updated: 2024-04-15

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm # Optional: for progress bar
from utils import logger # Import project logger
from .model import CBOW # Import model definition

def train_epoch(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int # For logging
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (CBOW): The CBOW model instance.
        dataloader (DataLoader): DataLoader providing context-target batches.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): The optimizer (e.g., Adam).
        device (torch.device): The device to train on (CPU or MPS).
        epoch (int): Current epoch number for logging.

    Returns:
        float: The average loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    # Optional: Progress bar
    # Use tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}") if total_epochs is passed
    data_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

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

        # Update progress bar description (optional)
        if batch_idx % 10 == 0: # Update less frequently
             data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


def train_model(
    model: CBOW,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    save_path: str = "models/word2vec" # Directory to save model
):
    """
    Orchestrates the model training over multiple epochs.

    Args:
        model (CBOW): The model to train.
        dataloader (DataLoader): The data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs to train.
        save_path (str): Directory path to save the trained model state.
    """
    logger.info(f"🚀 Starting CBOW training for {epochs} epochs on device: {device.type.upper()}")
    model.to(device) # Ensure model is on the correct device

    for epoch in range(epochs):
        avg_epoch_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
        logger.info(f"✅ Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model's state dictionary
    logger.info("🏁 Training finished.")
    try:
        os.makedirs(save_path, exist_ok=True)
        model_save_file = os.path.join(save_path, "cbow_embeddings.pth")
        # Save only the embeddings layer's state_dict if that's all needed later
        # torch.save(model.embeddings.state_dict(), model_save_file)
        # Or save the whole model state
        torch.save(model.state_dict(), model_save_file)
        logger.info(f"💾 Model state saved to: {model_save_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save model state: {e}", exc_info=True)