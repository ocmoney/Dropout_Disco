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
    def __init__(self, vocab_path, cbow_model_path, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
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

    def forward(self, x):
        # x is already embedded and averaged from the collate function
        return self.regression_model(x)


class TextDataset(Dataset):
    def __init__(self, texts, targets, vocab):
        """
        Custom Dataset for text regression.
        
        Args:
            texts (List[str]): List of input texts.
            targets (List[float]): List of target regression values.
            vocab (Vocabulary): Vocabulary object for tokenization.
        """
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        # Convert text to lowercase and split into tokens
        tokens = text.lower().split()
        # Get indices for each token, handling unknown words
        indices = [self.vocab.get_index(token) for token in tokens]
        return torch.tensor(indices, dtype=torch.long), target


def make_collate_fn(model):
    def collate_fn(batch):
        # Separate the sequences and targets
        sequences, targets = zip(*batch)
        
        # Convert targets to tensor
        targets = torch.stack(targets)
        
        # Process each sequence through the model's embedding layer
        embedded_sequences = []
        for seq in sequences:
            # Get embeddings for the sequence
            embeddings = model.embedding(seq)
            # Average the embeddings
            avg_embedding = embeddings.mean(dim=0)
            embedded_sequences.append(avg_embedding)
        
        # Stack the averaged embeddings
        embedded_batch = torch.stack(embedded_sequences)
        
        return embedded_batch, targets
    return collate_fn


def train_model(model, dataloader, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss = 0
    batch_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    # Print initial message
    logger.info(f"Starting epoch {epoch_num} with {len(dataloader)} batches")
    
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
        batch_count += 1
        
        # Check time after each batch
        current_time = time.time()
        elapsed_since_last_print = current_time - last_print_time
        
        # Print progress every minute (60 seconds)
        if elapsed_since_last_print >= 60:
            elapsed_time = current_time - start_time
            avg_loss = total_loss / batch_count
            progress = (batch_count / len(dataloader)) * 100
            remaining_batches = len(dataloader) - batch_count
            
            logger.info(f"Epoch {epoch_num} | Progress: {progress:.2f}% | Remaining batches: {remaining_batches} | Avg Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
            last_print_time = current_time
    
    # Print final message for this epoch
    final_avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch_num} completed. Final average loss: {final_avg_loss:.4f}")
    
    return final_avg_loss


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def predict_score(model, text, device):
    """
    Predict score for a single text input.
    
    Args:
        model (TextToRegressionModel): Trained model
        text (str): Input text to predict score for
        device (str): Device to run prediction on
        
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


def prepare_data(data):
    """
    Prepare data for training.
    
    Args:
        data (pandas.DataFrame): Raw Hacker News data
        
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
        vocab=model.vocab
    )
    
    test_dataset = TextDataset(
        texts=test_df['title'].tolist(),
        targets=test_df['score'].tolist(),
        vocab=model.vocab
    )
    
    # Create dataloaders
    batch_size = 32
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


def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    """
    Train and evaluate the model.
    
    Args:
        model (TextToRegressionModel): Model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tuple: (trained_model, training_losses, test_losses)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    # Training loop
    training_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch+1)
        training_losses.append(train_loss)
        
        # Evaluate
        test_loss = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return model, training_losses, test_losses


def main():
    """Main function to run the script."""
    # Fetch data
    data = fetch_hacker_news_data()
    
    # Prepare data
    model, train_loader, test_loader = prepare_data(data)
    
    # Train and evaluate model
    trained_model, training_losses, test_losses = train_and_evaluate(
        model, train_loader, test_loader, num_epochs=10
    )
    
    # Example prediction
    sample_text = "The best way to learn programming"
    predicted_score = predict_score(trained_model, sample_text, 
                                   torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Predicted score for '{sample_text}': {predicted_score:.2f}")


if __name__ == "__main__":
    main() 