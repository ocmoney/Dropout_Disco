import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ..word2vec.vocabulary import Vocabulary

# filepath: src/regression/TakeEmbedToHN.py

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        """
        A simple feedforward regression model.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int, optional): Dimension of the hidden layer. If None, no hidden layer is used.
        """
        super().__init__()
        if hidden_dim:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Output a single regression value
            )
        else:
            self.model = nn.Linear(input_dim, 1)  # Direct regression without hidden layers

    def forward(self, x):
        """
        Forward pass for the regression model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Regression output.
        """
        return self.model(x)


class TextToRegressionModel(nn.Module):
    def __init__(self, vocab_path, cbow_model_path, input_dim, hidden_dim=None):
        """
        Combines vocabulary, CBOW embeddings, and regression model.

        Args:
            vocab_path (str): Path to the saved vocabulary JSON.
            cbow_model_path (str): Path to the saved CBOW model state.
            input_dim (int): Dimension of the input embeddings.
            hidden_dim (int, optional): Hidden layer size for regression model.
        """
        super().__init__()
        # Load vocabulary
        self.vocab = Vocabulary.load_vocab(vocab_path)

        # Load CBOW model and extract embedding layer
        cbow_state = torch.load(cbow_model_path, map_location=torch.device('cpu'))
        self.embedding = nn.Embedding.from_pretrained(cbow_state['embedding.weight'])

        # Initialize regression model
        self.regression_model = RegressionModel(input_dim, hidden_dim)

    def preprocess(self, text):
        """
        Preprocesses text: lowercases, splits, and tokenizes.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Tokenized indices.
        """
        tokens = text.lower().split()
        indices = [self.vocab.get_index(token) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, text):
        """
        Forward pass: preprocess, embed, average, and predict.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Regression model output.
        """
        # Preprocess text
        token_indices = self.preprocess(text)

        # Embed and average
        embeddings = self.embedding(token_indices)
        avg_embedding = embeddings.mean(dim=0)

        # Pass to regression model
        return self.regression_model(avg_embedding)
    


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
        tokens = [self.vocab.get_index(token) for token in text.lower().split()]
        return torch.tensor(tokens, dtype=torch.long), target


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


# Example usage
if __name__ == "__main__":
    # Paths
    vocab_path = r"models\word2vec\text8_vocab.json"
    cbow_model_path = "models/word2vec/cbow_model_state.pth"

    #     # Build absolute paths
    # vocab_path = os.path.join(root_dir, "models", "text8_vocab.json")
    # cbow_model_path = os.path.join(root_dir, "models", "cbow_model_state.pth")

    # Hyperparameters
    input_dim = 128  # Embedding size
    hidden_dim = 128
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TextToRegressionModel(vocab_path, cbow_model_path, input_dim, hidden_dim).to(device)

    # Load data
    texts = ["example sentence one", "another example sentence"]  # Replace with your data
    targets = [1.0, 2.0]  # Replace with your target values
    vocab = model.vocab  # Use the same vocabulary as the model
    dataset = TextDataset(texts, targets, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluation
    eval_loss = evaluate_model(model, dataloader, criterion, device)
    print(f"Evaluation Loss: {eval_loss:.4f}")