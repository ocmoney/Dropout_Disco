import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model_test import TextToRegressionModel, TextDataset
import os

def make_collate_fn(model):
    """
    Creates a collate function for batching variable length sequences.
    
    Args:
        model: The model instance (used to access vocabulary)
    
    Returns:
        callable: A collate function that handles padding
    """
    def collate_fn(batch):
        # Separate texts and targets
        texts, targets = zip(*batch)
        
        # Get lengths of each sequence
        lengths = [len(text.split()) for text in texts]
        max_len = max(lengths)
        
        # Create padded tensor for texts
        padded_indices = []
        for text in texts:
            # Tokenize text
            tokens = text.lower().split()
            indices = [model.vocab.get_index(token) for token in tokens]
            
            # Pad sequence
            padding = [model.vocab.get_index(model.vocab.unk_token)] * (max_len - len(indices))
            padded_indices.append(indices + padding)
        
        # Convert to tensors
        text_tensor = torch.tensor(padded_indices, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.float)
        
        return text_tensor, target_tensor
    
    return collate_fn

def prepare_datasets(data_path):
    """
    Prepare train and test datasets from data file.
    Expected format: each line contains "text<tab>score"
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    texts = []
    scores = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, score = line.strip().split('\t')
            texts.append(text)
            scores.append(float(score))
    
    # Split into train and test (80/20)
    split_idx = int(len(texts) * 0.8)
    
    train_texts = texts[:split_idx]
    train_scores = scores[:split_idx]
    test_texts = texts[split_idx:]
    test_scores = scores[split_idx:]
    
    return train_texts, train_scores, test_texts, test_scores

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vocab_path = os.path.join(base_dir, "models/word2vec/text8_vocab_NW10M_MF5.json")
    cbow_path = os.path.join(base_dir, "models/word2vec/CBOW_D128_W3_NW10M_MF5_E5_LR0.001_BS512/model_state.pth")
    data_path = os.path.join(base_dir, "data/hn_scores.txt")  # You'll need to create this

    # Initialize model
    model = TextToRegressionModel(
        vocab_path=vocab_path,
        cbow_model_path=cbow_path,
        input_dim=128,
        hidden_dim=128
    )
    model = model.to(device)

    # Prepare datasets
    train_texts, train_scores, test_texts, test_scores = prepare_datasets(data_path)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_scores, model.vocab)
    test_dataset = TextDataset(test_texts, test_scores, model.vocab)

    # Set up DataLoaders
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

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (texts, targets) in enumerate(train_loader):
            texts, targets = texts.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for texts, targets in test_loader:
                texts, targets = texts.to(device), targets.to(device)
                outputs = model(texts)
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.4f}')

if __name__ == "__main__":
    main() 