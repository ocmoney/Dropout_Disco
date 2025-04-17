#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OC_Collect_EvenScore.py - Script for collecting and processing Hacker News data
and training a text-to-regression model with balanced score distribution.
"""

import os
import sys
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import time
import json
import numpy as np
import wandb
import torch.optim as optim
from collections import defaultdict

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import project modules
from word2vec.vocabulary import Vocabulary
from utils import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.info("Starting OC_Collect_EvenScore.py script")

# Database configuration
DB_URI = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(DB_URI)

class TextToRegressionModel(nn.Module):
    def __init__(self, vocab_path, cbow_model_path, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Combines vocabulary, CBOW embeddings, and MLP regression model.
        
        Args:
            vocab_path (str): Path to the saved vocabulary JSON.
            cbow_model_path (str): Path to the saved CBOW model state.
            input_dim (int): Dimension of the input embeddings.
            hidden_dims (List[int]): List of hidden layer dimensions.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Load vocabulary
        self.vocab = Vocabulary.load_vocab(vocab_path)
        
        # Load CBOW model and extract embedding layer
        cbow_state = torch.load(cbow_model_path, map_location=torch.device('cpu'))
        self.embedding = nn.Embedding.from_pretrained(cbow_state['embeddings.weight'])
        
        # Enable gradient updates for the embedding layer
        self.embedding.weight.requires_grad = True
        
        # Initialize MLP layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Add final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine all layers
        self.regression_model = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias to avoid zero outputs

    def forward(self, x):
        # x is already embedded and averaged from the collate function
        return self.regression_model(x)


class TextDataset(Dataset):
    def __init__(self, texts, targets, vocab, log_transform=True):
        """
        Custom Dataset for text regression.
        
        Args:
            texts (List[str]): List of input texts.
            targets (List[float]): List of target regression values.
            vocab (Vocabulary): Vocabulary object for tokenization.
            log_transform (bool): Whether to apply log transformation to targets.
        """
        self.texts = texts
        self.log_transform = log_transform
        
        # Apply log transformation to targets if specified
        if log_transform:
            # Add a small constant to avoid log(0)
            self.targets = torch.tensor(np.log1p(targets), dtype=torch.float32)
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)
            
        self.vocab = vocab
        
        # Pre-process texts to remove unknown words
        self.processed_texts = []
        for text in texts:
            # Convert text to lowercase and split into tokens
            tokens = text.lower().split()
            
            # Filter out unknown words
            filtered_tokens = [token for token in tokens if self.vocab.get_index(token) != self.vocab.get_index("<unk>")]
            
            # If all words were unknown, use a single <unk> token to avoid empty sequences
            if not filtered_tokens:
                filtered_tokens = ["<unk>"]
                
            self.processed_texts.append(filtered_tokens)
            
        # Log statistics about unknown words
        total_tokens = sum(len(text.lower().split()) for text in texts)
        unknown_tokens = sum(1 for text in texts for token in text.lower().split() 
                            if self.vocab.get_index(token) == self.vocab.get_index("<unk>"))
        logger.info(f"Removed {unknown_tokens} unknown tokens out of {total_tokens} total tokens ({unknown_tokens/total_tokens*100:.2f}%)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Use pre-processed tokens instead of processing on-the-fly
        tokens = self.processed_texts[idx]
        target = self.targets[idx]
        
        # Get indices for each token
        indices = [self.vocab.get_index(token) for token in tokens]
        return torch.tensor(indices, dtype=torch.long), target


def make_collate_fn(model):
    def collate_fn(batch):
        # Separate the sequences and targets
        sequences, targets = zip(*batch)
        
        # Convert targets to tensor
        targets = torch.stack(targets)
        
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Process each sequence through the model's embedding layer
        embedded_sequences = []
        for seq in sequences:
            # Move sequence to the same device as the model
            seq = seq.to(device)
            # Get embeddings for the sequence
            embeddings = model.embedding(seq)
            # Average the embeddings
            avg_embedding = embeddings.mean(dim=0)
            embedded_sequences.append(avg_embedding)
        
        # Stack the averaged embeddings
        embedded_batch = torch.stack(embedded_sequences)
        
        # Move targets to the same device
        targets = targets.to(device)
        
        return embedded_batch, targets
    return collate_fn


class BalancedWeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Base weights start at 1.0
        weights = torch.ones_like(target)
        
        # Weight boundaries matching exact percentiles from the data
        weights[target > 2] = 2.0     # ~30th percentile
        weights[target > 3] = 3.0     # ~55th percentile
        weights[target > 4] = 4.0     # ~70th percentile
        weights[target > 5] = 5.0     # ~75th percentile
        weights[target > 8] = 10.0    # ~80th percentile
        weights[target > 17] = 20.0   # ~85th percentile
        weights[target > 41] = 40.0   # ~90th percentile
        weights[target > 105] = 80.0  # ~95th percentile
        weights[target > 167] = 160.0 # ~97th percentile
        weights[target > 329] = 320.0 # ~99th percentile
        weights[target > 1000] = 640.0 # For extreme outliers
        
        # For scores ≤ 3 (majority of data), use simple weighted absolute error
        mask_low = target <= 3
        loss_low = weights[mask_low] * torch.abs(pred[mask_low] - target[mask_low])
        
        # For higher scores (> 3), combine absolute and relative error
        mask_high = target > 3
        abs_diff = torch.abs(pred[mask_high] - target[mask_high])
        relative_error = abs_diff / target[mask_high]
        
        # Stronger penalty for underestimation of high scores
        underestimation_penalty = torch.ones_like(relative_error)
        underestimation_mask = pred[mask_high] < target[mask_high]
        underestimation_penalty[underestimation_mask] = 1.5  # 50% extra penalty for underestimation
        
        # Additional multiplier for very high scores (> 100)
        high_score_multiplier = torch.ones_like(relative_error)
        very_high_mask = target[mask_high] > 100
        high_score_multiplier[very_high_mask] = 2.0  # Double the penalty for very high scores
        
        loss_high = weights[mask_high] * (
            0.3 * abs_diff + 
            0.7 * target[mask_high] * relative_error * underestimation_penalty * high_score_multiplier
        )
        
        # Combine losses
        total_loss = torch.zeros_like(target)
        total_loss[mask_low] = loss_low
        total_loss[mask_high] = loss_high
        
        return torch.mean(total_loss)


def train_model(model, dataloader, optimizer, criterion, device, use_wandb=True):
    model.train()
    total_loss = 0
    start_time = time.time()
    last_print_time = start_time
    batch_counter = 0
    logger.info(f"Starting training on {len(dataloader)} batches")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_counter += 1
        
        # Log batch-level metrics to wandb if enabled
        if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid too much data
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx,
                "global_step": batch_idx
            })
        
        elapsed_since_last_print = time.time() - last_print_time
        if elapsed_since_last_print >= 60:
            current_time = time.time() - start_time
            avg_loss = total_loss / batch_counter
            progress = (batch_counter / len(dataloader)) * 100
            logger.info(f"Training progress: {progress:.2f}% | Avg Loss: {avg_loss:.4f} | Time: {current_time:.2f}s")
            last_print_time = time.time()
    
    logger.info(f"Epoch completed. Final average loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device, log_transform=True):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model (TextToRegressionModel): Model to evaluate
        dataloader (DataLoader): DataLoader containing evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to run evaluation on
        log_transform (bool): Whether targets were log-transformed
        
    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0
    total_original_loss = 0  # For tracking loss in original scale
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss in transformed space
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            
            # Calculate loss in original scale if log transformation was used
            if log_transform:
                # Convert predictions back to original scale
                original_outputs = torch.exp(outputs.squeeze()) - 1
                original_targets = torch.exp(targets) - 1
                original_loss = nn.L1Loss()(original_outputs, original_targets)
                total_original_loss += original_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # If log transformation was used, also calculate and log the original scale loss
    if log_transform:
        avg_original_loss = total_original_loss / len(dataloader)
        logger.info(f"Test Loss (log scale): {avg_loss:.4f}, Test Loss (original scale): {avg_original_loss:.4f}")
        return avg_loss, avg_original_loss
    
    return avg_loss


def predict_score(model, text, device, log_transform=True):
    """
    Predict score for a single text input.
    
    Args:
        model (TextToRegressionModel): Trained model
        text (str): Input text to predict score for
        device (str): Device to run prediction on
        log_transform (bool): Whether targets were log-transformed
        
    Returns:
        float: Predicted score
    """
    model.eval()
    with torch.no_grad():
        # Preprocess the text
        tokens = text.lower().split()
        indices = [model.vocab.get_index(token) for token in tokens]
        token_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Get embeddings and average
        embeddings = model.embedding(token_tensor)
        avg_embedding = embeddings.mean(dim=1)
        
        # Get prediction
        prediction = model.regression_model(avg_embedding)
        
        # If log transformation was used, apply inverse transformation
        if log_transform:
            # Apply exp(x) - 1 to get back to original scale
            prediction = torch.exp(prediction) - 1
            
        return prediction.item()


def fetch_hacker_news_data():
    """
    Fetch Hacker News data from the database.
    
    Returns:
        pandas.DataFrame: DataFrame containing Hacker News stories
    """
    logger.info("Fetching Hacker News data from database")
    
    # Query the database
    res = pd.read_sql("""
        SELECT *
        FROM "hacker_news"."items" a
        WHERE a.type = 'story'
            AND a.time >= '2023-01-01 00:00:00'
            AND a.dead IS NOT TRUE
            AND LENGTH(a.title) > 0
    """, engine)
    
    logger.info(f"Fetched {len(res)} stories from Hacker News")
    return res


def create_balanced_dataset(data, num_percentiles=10, samples_per_percentile=1000):
    """
    Create a balanced dataset by sampling an equal number of titles from each score percentile.
    
    Args:
        data (pandas.DataFrame): Raw Hacker News data
        num_percentiles (int): Number of percentiles to divide the score range into
        samples_per_percentile (int): Number of samples to take from each percentile
        
    Returns:
        pandas.DataFrame: Balanced dataset
    """
    logger.info(f"Creating balanced dataset with {num_percentiles} percentiles and {samples_per_percentile} samples per percentile")
    
    # Extract titles and scores
    titles_and_scores = data.loc[:, ['title', 'score']].copy()
    
    # Calculate percentile boundaries
    percentiles = np.linspace(0, 100, num_percentiles + 1)
    percentile_values = np.percentile(titles_and_scores['score'], percentiles)
    
    # Create a balanced dataset
    balanced_data = []
    
    for i in range(num_percentiles):
        lower_bound = percentile_values[i]
        upper_bound = percentile_values[i + 1]
        
        # Get titles in this percentile range
        percentile_data = titles_and_scores[
            (titles_and_scores['score'] >= lower_bound) & 
            (titles_and_scores['score'] < upper_bound)
        ]
        
        # Sample titles from this percentile
        if len(percentile_data) > 0:
            sampled_data = percentile_data.sample(
                min(samples_per_percentile, len(percentile_data)), 
                replace=len(percentile_data) < samples_per_percentile
            )
            balanced_data.append(sampled_data)
            logger.info(f"Percentile {i+1}: {lower_bound:.2f} - {upper_bound:.2f}, sampled {len(sampled_data)} titles")
        else:
            logger.warning(f"No titles found in percentile {i+1}: {lower_bound:.2f} - {upper_bound:.2f}")
    
    # Combine all sampled data
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    logger.info(f"Created balanced dataset with {len(balanced_df)} titles")
    
    # Display score distribution of the balanced dataset
    display_score_statistics(balanced_df, title="Balanced Dataset Score Statistics")
    
    return balanced_df


def prepare_data(data, log_transform=True):
    """
    Prepare data for training.
    
    Args:
        data (pandas.DataFrame): Raw Hacker News data
        log_transform (bool): Whether to apply log transformation to targets
        
    Returns:
        tuple: (model, train_loader, test_loader)
    """
    # Create a balanced dataset
    balanced_data = create_balanced_dataset(data)
    
    # Split the data
    train_df, test_df = train_test_split(balanced_data, test_size=0.2, random_state=42)
    
    # Initialize model to get vocabulary
    model = TextToRegressionModel(
        vocab_path="models/word2vec/text8_vocab_NW10M_MF5.json",
        cbow_model_path="models/word2vec/CBOW_D128_W3_NW10M_MF5_E5_LR0.001_BS512/model_state.pth",
        input_dim=128
    )
    
    # Create datasets
    train_dataset = TextDataset(
        texts=train_df['title'].tolist(),
        targets=train_df['score'].tolist(),
        vocab=model.vocab,
        log_transform=log_transform
    )
    
    test_dataset = TextDataset(
        texts=test_df['title'].tolist(),
        targets=test_df['score'].tolist(),
        vocab=model.vocab,
        log_transform=log_transform
    )
    
    # Create dataloaders
    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(model)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(model)
    )
    
    return model, train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, num_epochs, learning_rate, device, use_wandb=True):
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="text-to-regression-balanced",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "model": model.__class__.__name__,
                "optimizer": "Adam",
                "loss": "BalancedWeightedL1Loss"  # Updated loss name
            }
        )
        wandb.watch(model, log="all")
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BalancedWeightedL1Loss()  # Use the new loss function
    
    # Training loop
    training_losses = []
    test_losses = []
    test_losses_original = []
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, criterion, device, use_wandb)
        training_losses.append(train_loss)
        
        # Evaluate
        test_result = evaluate_model(model, test_loader, criterion, device)
        
        # Handle the case where evaluate_model returns a tuple (when log_transform=True)
        if isinstance(test_result, tuple):
            test_loss, test_loss_original = test_result
            test_losses.append(test_loss)
            test_losses_original.append(test_loss_original)
        else:
            test_loss = test_result
            test_losses.append(test_loss)
        
        # Log metrics to wandb if enabled
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # Close wandb run if it was initialized
    if use_wandb:
        wandb.finish()
    
    return training_losses, test_losses


def save_model(model, training_losses, test_losses, test_losses_original=None, log_transform=True):
    """
    Save the trained model and training history.
    
    Args:
        model (TextToRegressionModel): Trained model
        training_losses (List[float]): List of training losses
        test_losses (List[float]): List of test losses
        test_losses_original (List[float], optional): List of test losses in original scale
        log_transform (bool): Whether targets were log-transformed
    """
    # Create directory if it doesn't exist
    save_dir = "models/text_to_regression_balanced"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save model state
    model_path = os.path.join(save_dir, f"model_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'log_transform': log_transform
    }, model_path)
    
    # Save training history
    history = {
        "training_losses": training_losses,
        "test_losses": test_losses,
        "log_transform": log_transform
    }
    
    # Add original scale losses if available
    if test_losses_original is not None:
        history["test_losses_original"] = test_losses_original
        
    history_path = os.path.join(save_dir, f"history_{timestamp}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Training history saved to {history_path}")


def display_sample_titles(data, num_samples=50):
    """
    Get a sample of title texts from the dataset.
    
    Args:
        data (pandas.DataFrame): Dataset containing title texts
        num_samples (int): Number of samples to display
        
    Returns:
        List[str]: List of sample titles
    """
    return data['title'].sample(num_samples).tolist()


def display_training_titles(model, sample_titles):
    """
    Display how the sample titles are processed for training.
    
    Args:
        model (TextToRegressionModel): The model with vocabulary
        sample_titles (List[str]): List of sample titles to process
    """
    logger.info("Displaying sample titles and their processed versions:")
    for i, title in enumerate(sample_titles, 1):
        # Process the title the same way as in TextDataset.__getitem__
        tokens = title.lower().split()
        # Get indices for each token
        indices = [model.vocab.get_index(token) for token in tokens]
        # Count unknown tokens
        unk_count = sum(1 for idx in indices if idx == model.vocab.get_index("<unk>"))
        
        # Create a processed version showing UNK tokens
        processed_tokens = []
        for token in tokens:
            idx = model.vocab.get_index(token)
            if idx == model.vocab.get_index("<unk>"):
                processed_tokens.append("<unk>")
            else:
                processed_tokens.append(token)
        processed_text = " ".join(processed_tokens)
        
        logger.info(f"{i}. Original: '{title}'")
        logger.info(f"   Processed: '{processed_text}' | Tokens: {len(tokens)} | UNK tokens: {unk_count}")
    logger.info("")


def display_score_statistics(data, title="Score Statistics"):
    """
    Calculate and display statistics for the scores of all titles.
    
    Args:
        data (pandas.DataFrame): DataFrame containing Hacker News stories
        title (str): Title for the statistics display
    """
    scores = data['score'].values
    
    # Calculate statistics
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    range_score = max_score - min_score
    std_score = np.std(scores)
    
    # Calculate IQR
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr_score = q3 - q1
    
    # Display statistics
    logger.info(title)
    logger.info(f"  Mean: {mean_score:.2f}")
    logger.info(f"  Range: {min_score:.2f} to {max_score:.2f} (width: {range_score:.2f})")
    logger.info(f"  Standard Deviation: {std_score:.2f}")
    logger.info(f"  IQR: {iqr_score:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")
    
    # Calculate percentiles with finer granularity in the tail
    # Regular percentiles from 0 to 95 in steps of 5
    regular_percentiles = list(range(0, 96, 5))
    # Fine-grained percentiles from 95 to 100 in steps of 1
    tail_percentiles = list(range(96, 101))
    # Combine both lists
    percentiles = regular_percentiles + tail_percentiles
    
    percentile_values = {p: np.percentile(scores, p) for p in percentiles}
    
    # Print percentiles in a table format
    logger.info("\nPercentile Distribution:")
    logger.info("Percentile | Actual Score")
    logger.info("-" * 30)
    for p in percentiles:
        logger.info(f"{p:3d}th     | {percentile_values[p]:.2f}")
    
    # Count of scores in different ranges
    logger.info("\nScore Distribution:")
    logger.info(f"  Scores <= 1: {np.sum(scores <= 1)} ({np.sum(scores <= 1)/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 1 and <= 10: {np.sum((scores > 1) & (scores <= 10))} ({np.sum((scores > 1) & (scores <= 10))/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 10 and <= 100: {np.sum((scores > 10) & (scores <= 100))} ({np.sum((scores > 10) & (scores <= 100))/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 100: {np.sum(scores > 100)} ({np.sum(scores > 100)/len(scores)*100:.2f}%)")


def setup_logging():
    """Set up logging configuration for the application."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s',
        handlers=[
            logging.FileHandler("logs/dropout_disco_balanced.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging system initialized successfully!")


def load_data():
    """
    Load and preprocess data for training.
    
    Returns:
        pandas.DataFrame: Processed data ready for training
    """
    logger.info("Loading data...")
    
    # Fetch data from database
    data = fetch_hacker_news_data()
    if data is None or len(data) == 0:
        logger.error("Failed to fetch data or data is empty")
        return None
    
    # Display score statistics
    display_score_statistics(data, title="Original Dataset Score Statistics")
    
    # Display sample titles before preparation
    sample_titles = display_sample_titles(data)
    
    logger.info(f"Loaded {len(data)} records successfully")
    return data


def main():
    # Set up logging
    setup_logging()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    data = load_data()
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Prepare data for training
    model, train_loader, test_loader = prepare_data(data)
    if train_loader is None or test_loader is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Move model to device
    model = model.to(device)
    
    # Train and evaluate model with wandb logging enabled
    training_losses, test_losses = train_and_evaluate(
        model, train_loader, test_loader, num_epochs=5, learning_rate=0.0005, device=device, use_wandb=True
    )
    
    # Save the model
    save_model(model, training_losses, test_losses)
    
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
