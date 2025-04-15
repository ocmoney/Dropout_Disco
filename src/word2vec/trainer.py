# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/word2vec/trainer.py
# Description: Training loop logic for Word2Vec models.
# Created: 2025-04-15
# Updated: 2025-04-15

import torch
from torch.utils.data import DataLoader, Dataset
# Add imports for your model, data utils, optimizer etc.

# TODO: Define a PyTorch Dataset for your training pairs and negative sampling
class Word2VecDataset(Dataset):
     def __init__(self, pairs, vocab_size, num_negative_samples=5): # Add necessary params
         self.pairs = pairs
         self.vocab_size = vocab_size
         self.num_negative_samples = num_negative_samples
         # Precompute negative sampling distribution if needed
         # self.neg_sampling_weights = neg_sampling_weights

     def __len__(self):
         return len(self.pairs)

     def __getitem__(self, idx):
         center_word, context_word = self.pairs[idx]
         # Sample negative words
         # TODO: Implement actual negative sampling based on distribution
         negative_samples = torch.randint(0, self.vocab_size, (self.num_negative_samples,))
         # Avoid sampling the true context word (less critical but good practice)
         
         return torch.tensor(center_word), torch.tensor(context_word), negative_samples


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        center, context, negatives = [item.to(device) for item in batch]

        optimizer.zero_grad()
        loss = model(center, context, negatives)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 1000 == 0: # Print progress every 1000 batches
             print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def train_word2vec_model(model, dataset, optimizer, device, epochs=1, batch_size=512):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    print(f"\n--- Starting Word2Vec Training ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Device: {device}")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("\n--- Training Finished ---")

