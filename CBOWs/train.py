from tokeniser import Tokeniser
from model import CROW as Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
import itertools
import numpy as np

# Add this after imports and before hyperparameters
def collate_batch(batch):
    """Collate function to properly batch the data"""
    contexts = []
    targets = []
    
    for context, target in batch:
        contexts.append(context.long())  # Convert to long here
        targets.append(target.long())    # Convert to long here
    
    # Stack tensors into batches
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    
    return contexts, targets

# Hyperparameters
CONTEXT_SIZE = 2
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

class TokenBatchDataset(IterableDataset):
    def __init__(self, tokenizer, context_size, chunk_size=1000000):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.chunk_size = chunk_size
        self.path = Path('./CBOWs/text8').absolute()
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.path, 'r') as f:
            while True:
                # Read text chunk by chunk
                text_chunk = f.read(self.chunk_size)
                if not text_chunk:
                    break
                    
                # Process chunk
                tokens = self.tokenizer.getIdentifyer(text_chunk.lower())
                
                # Create sliding windows over tokens
                for i in range(self.context_size, len(tokens) - self.context_size):
                    context = tokens[i-self.context_size:i] + \
                             tokens[i+1:i+self.context_size+1]
                    target = tokens[i]
                    
                    # Use int32 instead of int64 to save memory
                    yield (
                        torch.tensor(context, dtype=torch.int32),
                        torch.tensor(target, dtype=torch.int32)
                    )

def train():
    print("Initializing...")
    tokeniser = Tokeniser()
    model = Model(tokeniser.getCorpusSize())
    
    print("Creating dataset...")
    dataset = TokenBatchDataset(tokeniser, CONTEXT_SIZE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_batch  # Add custom collate function
    )
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (contexts, targets) in enumerate(dataloader):
            # Ensure proper batch dimensions
            if contexts.size(0) != BATCH_SIZE:
                continue
                
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                log_probs = model(contexts)
                
                # Debug print
                print(f"Batch shapes - contexts: {contexts.shape}, "
                      f"log_probs: {log_probs.shape}, targets: {targets.shape}")
                
                # Compute loss
                loss = criterion(log_probs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'./CBOWs/checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), './CBOWs/cbow_model.pth')
    print("Training completed!")

if __name__ == "__main__":
    train()