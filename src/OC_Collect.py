#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OC_Collect.py - Script for collecting and processing Hacker News data
and training a text-to-regression model.
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

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import project modules
from word2vec.vocabulary import Vocabulary
from utils import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.info("Starting OC_Collect.py script")

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


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    last_print_time = start_time
    batch_counter = 0
    logger.info(f"Starting training on {len(dataloader)} batches")
    for inputs, targets in dataloader:
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


def prepare_data(data, log_transform=True):
    """
    Prepare data for training.
    
    Args:
        data (pandas.DataFrame): Raw Hacker News data
        log_transform (bool): Whether to apply log transformation to targets
        
    Returns:
        tuple: (train_dataset, test_dataset, train_loader, test_loader)
    """
    # Extract titles and scores
    titles_and_scores = data.loc[:, ['title', 'score']].copy()
    
    # Split the data
    train_df, test_df = train_test_split(titles_and_scores, test_size=0.2, random_state=42)
    
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


def train_and_evaluate(model, train_loader, test_loader, num_epochs=20, learning_rate=0.0005, log_transform=True):
    """
    Train and evaluate the model.
    
    Args:
        model (TextToRegressionModel): Model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        log_transform (bool): Whether targets were log-transformed
        
    Returns:
        tuple: (trained_model, training_losses, test_losses)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Use a more aggressive scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Use weighted L1 loss (MAE) for better handling of skewed distributions
    criterion = WeightedL1Loss()
    
    # Training loop
    training_losses = []
    test_losses = []
    test_losses_original = []  # For tracking original scale losses
    best_test_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        training_losses.append(train_loss)
        
        # Evaluate
        if log_transform:
            test_loss, test_loss_original = evaluate_model(model, test_loader, criterion, device, log_transform)
            test_losses.append(test_loss)
            test_losses_original.append(test_loss_original)
        else:
            test_loss = evaluate_model(model, test_loader, criterion, device, log_transform)
            test_losses.append(test_loss)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Check if this is the best model so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        if log_transform:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss (log): {test_loss:.4f}, Test Loss (original): {test_loss_original:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with test loss: {best_test_loss:.4f}")
    
    # Save the trained model
    if log_transform:
        save_model(model, training_losses, test_losses, test_losses_original, log_transform)
    else:
        save_model(model, training_losses, test_losses, log_transform)
    
    return model, training_losses, test_losses


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
    save_dir = "models/text_to_regression"
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


def display_score_statistics(data):
    """
    Calculate and display statistics for the scores of all titles.
    
    Args:
        data (pandas.DataFrame): DataFrame containing Hacker News stories
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
    logger.info("Score Statistics:")
    logger.info(f"  Mean: {mean_score:.2f}")
    logger.info(f"  Range: {min_score:.2f} to {max_score:.2f} (width: {range_score:.2f})")
    logger.info(f"  Standard Deviation: {std_score:.2f}")
    logger.info(f"  IQR: {iqr_score:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")
    
    # Additional percentiles for context
    p10 = np.percentile(scores, 10)
    p90 = np.percentile(scores, 90)
    logger.info(f"  10th Percentile: {p10:.2f}")
    logger.info(f"  90th Percentile: {p90:.2f}")
    
    # Count of scores in different ranges
    logger.info("Score Distribution:")
    logger.info(f"  Scores <= 1: {np.sum(scores <= 1)} ({np.sum(scores <= 1)/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 1 and <= 10: {np.sum((scores > 1) & (scores <= 10))} ({np.sum((scores > 1) & (scores <= 10))/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 10 and <= 100: {np.sum((scores > 10) & (scores <= 100))} ({np.sum((scores > 10) & (scores <= 100))/len(scores)*100:.2f}%)")
    logger.info(f"  Scores > 100: {np.sum(scores > 100)} ({np.sum(scores > 100)/len(scores)*100:.2f}%)")


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Calculate weights based on target values
        # Higher weights for extreme values
        weights = torch.ones_like(target)
        weights[target > 10] = 2.0
        weights[target > 50] = 5.0
        weights[target > 100] = 10.0
        
        # Apply weights to MSE loss
        return torch.mean(weights * (pred - target) ** 2)


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # More aggressive weighting for extreme values
        weights = torch.ones_like(target)
        weights[target > 5] = 3.0
        weights[target > 20] = 10.0
        weights[target > 50] = 20.0
        weights[target > 100] = 50.0
        
        # Add a small epsilon to avoid zero gradients
        return torch.mean(weights * torch.abs(pred - target) + 1e-6)


class ScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        # Ensure positive predictions with a minimum value
        return torch.clamp(x, min=0.1)


def main():
    """Main function to run the script."""
    # Fetch data
    data = fetch_hacker_news_data()
    
    # Display score statistics
    display_score_statistics(data)
    
    # Display sample titles before preparation
    sample_titles = display_sample_titles(data)
    
    # Prepare data with log transformation
    log_transform = True
    model, train_loader, test_loader = prepare_data(data, log_transform=log_transform)
    
    # Display how sample titles are processed for training
    display_training_titles(model, sample_titles)
    
    # Train and evaluate model
    trained_model, training_losses, test_losses = train_and_evaluate(
        model, train_loader, test_loader, num_epochs=5, log_transform=log_transform
    )
    
    # Example prediction
    sample_text = "The best way to learn programming"
    predicted_score = predict_score(trained_model, sample_text, 
                                   torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                   log_transform=log_transform)
    logger.info(f"Predicted score for '{sample_text}': {predicted_score:.2f}")


if __name__ == "__main__":
    main() 