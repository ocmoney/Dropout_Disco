# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: app/predictor.py
# Description: Handles loading models and making predictions.
# Created: 2025-04-15
# Updated: 2025-04-15

import torch
import numpy as np
# Add imports for your specific model classes if needed
# from src.regression.model import RegressionModel # Example
# from src.word2vec.model import Word2VecModel # Less likely needed directly

# --- Globals for loaded models (or use a class) ---
word_embeddings = None
vocab = None
regression_model = None
model_version = "0.0.0" # Should match main.py or be loaded

MODEL_DIR = "/code/models" # Path inside the Docker container

def load_models(w2v_model_path=f"{MODEL_DIR}/word2vec/word_embeddings.kv", # Or .pt etc
                vocab_path=f"{MODEL_DIR}/word2vec/text8_vocab.json",
                reg_model_path=f"{MODEL_DIR}/regression/hn_predictor_v{model_version}.pt"):
    """Loads the word embeddings, vocab, and regression model."""
    global word_embeddings, vocab, regression_model
    print(f"Attempting to load models...")
    try:
        # TODO: Implement actual loading based on how models were saved
        # Example for gensim KeyedVectors
        # from gensim.models import KeyedVectors
        # word_embeddings = KeyedVectors.load(w2v_model_path, mmap='r')

        # Example for vocab
        # import json
        # with open(vocab_path, 'r') as f:
        #     vocab = json.load(f)
        
        # Example for PyTorch regression model state_dict
        # model_state_dict = torch.load(reg_model_path)
        # Assuming you know the model architecture
        # input_dim = ... # e.g., embedding dimension
        # regression_model = RegressionModel(input_dim) # Instantiate your model class
        # regression_model.load_state_dict(model_state_dict)
        # regression_model.eval() # Set to evaluation mode

        print("Models loaded successfully (placeholder).") # Replace with actual success message
    except Exception as e:
        print(f"Error loading models: {e}")
        # Handle errors appropriately - maybe raise or exit

def predict(title: str) -> float:
    """Predicts the log score for a given title."""
    if regression_model is None or word_embeddings is None or vocab is None:
        # Optionally try loading here, or ensure loaded at startup
        print("Warning: Models not loaded. Returning default prediction.")
        # load_models() # Attempt loading if not already loaded
        # if regression_model is None: # Check again
        return 0.0 # Default prediction if models aren't ready

    # 1. Tokenize title using loaded vocab
    tokens = [token for token in title.lower().split() if token in vocab] # Simple split, needs alignment with training
    if not tokens:
        return np.log1p(1.0) # Predict median for score=2 if no known words

    # 2. Get embeddings and average
    # TODO: Implement based on how embeddings are stored
    # Example using gensim KV and handling UNK (assuming UNK is not in vocab)
    # vectors = [word_embeddings[token] for token in tokens if token in word_embeddings] # Check if token exists in embeddings
    # if not vectors:
    #     return np.log1p(1.0)
    # title_vector = np.mean(vectors, axis=0)

    # Placeholder vector
    embedding_dim = 128 # Match your actual dim
    title_vector = np.random.rand(embedding_dim) # Replace with actual averaging

    # 3. Predict using regression model
    try:
        # Convert numpy array to PyTorch tensor
        input_tensor = torch.tensor(title_vector, dtype=torch.float32).unsqueeze(0) # Add batch dimension

        with torch.no_grad(): # No need to track gradients during inference
            log_score_prediction = regression_model(input_tensor).item() # Get single scalar value

        # Optional: Inverse transform if needed (usually done after API return)
        # raw_score_prediction = np.expm1(log_score_prediction)

        return log_score_prediction # Return log score

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0.0 # Default on error

# Call load_models when this module is imported (or explicitly in main.py startup)
# load_models()

