
from sqlalchemy import create_engine
import pandas as pd

# --- Configuration ---
# Define the database URI directly
# !! In real projects, manage credentials securely (e.g., env variables, secrets manager) !!
DB_URI = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

engine = create_engine(DB_URI)
# --- Optional: Set up logging ---
import logging
logging.basicConfig(level=logging.INFO)


# Example: Show tables (PostgreSQL metadata)
res = pd.read_sql("""
    SELECT *
    FROM "hacker_news"."items" a
    WHERE a.type = 'story'
        AND a.time >= '2023-01-01 00:00:00'
        AND a.dead IS NOT TRUE
        AND LENGTH(a.title) > 0
        --LIMIT 10
""", engine)

res

titles_and_scores = res.loc[:, ['title', 'score']].copy()
titles_and_scores.head(50)
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from utils import logger  # Import the `logger` module from `utils`
vocab
import torch
import torch.nn as nn
from word2vec.vocabulary import Vocabulary as vocab
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
        self.vocab = vocab.load_vocab(vocab_path)
        
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
from torch.utils.data import Dataset, DataLoader

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

def make_collate_fn(model, device):
    def collate_fn(batch):
        # Separate the sequences and targets
        sequences, targets = zip(*batch)
        
        # Convert targets to tensor and move to the correct device
        targets = torch.stack(targets).to(device)
        
        # Process each sequence through the model's embedding layer
        embedded_sequences = []
        for seq in sequences:
            # Get embeddings for the sequence
            embeddings = model.embedding(seq.to(device))  # Move seq to the correct device
            # Average the embeddings
            avg_embedding = embeddings.mean(dim=0)
            embedded_sequences.append(avg_embedding)
        
        # Stack the averaged embeddings
        embedded_batch = torch.stack(embedded_sequences).to(device)
        
        return embedded_batch, targets
    return collate_fn
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
    return total_loss / len(dataloader)

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
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming titles_and_scores is your DataFrame
# Split the data
train_df, test_df = train_test_split(titles_and_scores, test_size=0.2, random_state=42)

# Create datasets
train_dataset = TextDataset(
    texts=train_df['title'].tolist(),
    targets=train_df['score'].tolist(),
    vocab=vocab  # Your existing vocabulary object
)

test_dataset = TextDataset(
    texts=test_df['title'].tolist(),
    targets=test_df['score'].tolist(),
    vocab=vocab
)
print(train_dataset)
# Detect the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize model
model = TextToRegressionModel(
    vocab_path="../models/word2vec/text8_vocab_NWAll_MF5.json",  # Replace with your actual path
    cbow_model_path="../models/word2vec/CBOW_D128_W5_NWAll_MF5_E15_LR0.001_BS512/model_state.pth",  # Replace with your actual path
    input_dim=128,  # Match your CBOW embedding dimension
)
model = model.to(device)

# Create datasets
train_dataset = TextDataset(
    texts=train_df['title'].tolist(),
    targets=train_df['score'].tolist(),
    vocab=model.vocab  # Use the model's vocabulary
)

test_dataset = TextDataset(
    texts=test_df['title'].tolist(),
    targets=test_df['score'].tolist(),
    vocab=model.vocab
)

# Create dataloaders with the custom collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=make_collate_fn(model, device)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=make_collate_fn(model, device)
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Evaluate model on test set
model.eval()
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs.squeeze(), targets).item()
        
        predictions.extend(outputs.squeeze().cpu().numpy())
        actuals.extend(targets.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# Calculate additional metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R2 Score: {r2:.4f}')

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
    model = model.to(device)
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
    
predict_score(model, "technology", device)
print(model)
res.head(20)

# find the entry in res df including the title "the best way to learn..."
# Search for the title in the res DataFrame
#matching_entries = res[res['title'].str.contains("", case=False, na=False)]
#print(matching_entries)



