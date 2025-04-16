import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict
import requests
from datetime import datetime, timedelta

class UpvotePredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super(UpvotePredictor, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HNDataset(Dataset):
    def __init__(self, titles: List[str], upvotes: List[int], word_embeddings: Dict[str, torch.Tensor]):
        self.titles = titles
        self.upvotes = upvotes
        self.word_embeddings = word_embeddings
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        words = title.lower().split()
        
        # Get embeddings for each word and average them
        embeddings = []
        for word in words:
            if word in self.word_embeddings:
                embeddings.append(self.word_embeddings[word])
        
        if not embeddings:  # If no words found in embeddings
            # Return zero vector of same dimension as embeddings
            embedding_dim = next(iter(self.word_embeddings.values())).shape[0]
            avg_embedding = torch.zeros(embedding_dim)
        else:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            
        return avg_embedding, torch.tensor(self.upvotes[idx], dtype=torch.float32)

def load_embeddings(model_path: str, vocab_path: str) -> Dict[str, torch.Tensor]:
    """Load word embeddings from the saved CBOW model."""
    # Load the model state
    model_state = torch.load(model_path)
    
    # Load vocabulary
    vocab = torch.load(vocab_path)
    
    # Create dictionary of word to embedding
    word_embeddings = {}
    for word, idx in vocab.word2idx.items():
        word_embeddings[word] = model_state['embeddings.weight'][idx]
    
    return word_embeddings

def fetch_hn_data(num_stories: int = 1000) -> tuple:
    """Fetch Hacker News stories and their upvotes."""
    # Using the Hacker News API
    base_url = "https://hacker-news.firebaseio.com/v0"
    
    # Get top stories
    response = requests.get(f"{base_url}/topstories.json")
    story_ids = response.json()[:num_stories]
    
    titles = []
    upvotes = []
    
    for story_id in story_ids:
        story = requests.get(f"{base_url}/item/{story_id}.json").json()
        if story and 'title' in story and 'score' in story:
            titles.append(story['title'])
            upvotes.append(story['score'])
    
    return titles, upvotes

def train_model(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                num_epochs: int = 10):
    """Train the upvote prediction model."""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

def main():
    # Load embeddings
    word_embeddings = load_embeddings('cbow_model_state.pth', 'vocab.pth')
    
    # Fetch Hacker News data
    titles, upvotes = fetch_hn_data()
    
    # Create dataset and dataloader
    dataset = HNDataset(titles, upvotes, word_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    embedding_dim = next(iter(word_embeddings.values())).shape[0]
    model = UpvotePredictor(embedding_dim)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, dataloader, criterion, optimizer)
    
    # Save the trained model
    torch.save(model.state_dict(), 'upvote_predictor.pth')

if __name__ == "__main__":
    main() 