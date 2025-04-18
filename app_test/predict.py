import torch
import torch.nn as nn
import numpy as np
import re

from word2vec.vocabulary import Vocabulary


# --------- Model Definition ---------
class TextToRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.regression_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.regression_model(x)


# --------- Load Vocabulary and Embedding Model ---------
vocab = Vocabulary.load_vocab('text8_vocab_NWAll_MF5.json')
cbow_model = torch.load('CBOW_D128_W5_NWAll_MF5_E15_LR0.001_BS512/model_state.pth', map_location='cpu')
cbow_emb = cbow_model['embeddings.weight']


# --------- Preprocessing + Embedding ---------
def preprocess_and_embed(text):
    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        tokens = text.lower().split()
        return tokens

    tokens = preprocess_text(text)
    token_indices = [vocab.get_index(token) for token in tokens]
    
    if not token_indices:
        return torch.zeros(cbow_emb.shape[1])  # embedding_dim

    token_tensor = torch.tensor(token_indices)
    with torch.no_grad():
        embeddings = cbow_emb[token_tensor]
    averaged_embedding = embeddings.mean(dim=0)
    return averaged_embedding


# --------- Prediction Function ---------
def predict_upvotes(text, model):
    text_embedding = preprocess_and_embed(text).unsqueeze(0)  # (1, emb_dim)
    with torch.no_grad():
        prediction = model(text_embedding)
    return np.exp(prediction.item()).item()



# --------- Main ---------
if __name__ == "__main__":
    
    
    try:
        # Load trained regression model
        model = torch.load('regression_model.pth', weights_only=False, map_location='cpu')
        model.eval()

        print("Model loaded. Ready to predict upvotes.")
        print("Press Ctrl+C to exit.")

        # Continuous prediction loop
        while True:
            user_input = input("Input a post title: ")
            print("Predicted upvotes:", predict_upvotes(user_input, model))

    except KeyboardInterrupt:
        print("\nExiting program.")












"""
# --------- Find Top Words ---------
def find_top_words(model, vocab, device):
    model.eval()
    model = model.to(device)
    word_scores = []
    for word in vocab.idx2word:  # Iterate through all words in the vocabulary
        preprocessed_embedding = preprocess_and_embed(word).unsqueeze(0).to(device)  # Preprocess and embed the word
        with torch.no_grad():
            prediction = model(preprocessed_embedding)  # Pass the embedding to the model
        word_scores.append((word, np.exp(prediction.item())))  # Store the word and its predicted score

    # Sort the words by their predicted scores in descending order
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return word_scores[:50]  # Return the top 50 words


# --------- Main ---------
if __name__ == "__main__":
    # Load the trained regression model
    model = torch.load('regression_model.pth', map_location='cpu', weights_only=False)
    model.eval()

    # Detect the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find the top 50 words
    top_words = find_top_words(model, vocab, device)
    print("Top 50 words with the highest predicted scores:")
    
    #print a [] list of just the top 50 words:
    top_words_list = [word for word, score in top_words]
    print(top_words_list)

"""