import json
import torch
import argparse
import os

def load_vocab(vocab_path):
    """Load vocabulary from a JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # Expecting vocab to be a dictionary mapping words to indices.
    return vocab

def extract_embeddings(vocab, state_dict, key='embedding.weight'):
    """Extract word embeddings from the state dictionary.
    
    If the key is not found, search for a candidate key that contains both 
    'embedding' and 'weight' (case-insensitive) and use that.
    """
    if key not in state_dict:
        print(f"Key '{key}' not found. Available keys: {list(state_dict.keys())}")
        candidate_keys = [k for k in state_dict.keys() 
                          if 'embedding' in k.lower() and 'weight' in k.lower()]
        if candidate_keys:
            print(f"Using candidate key: {candidate_keys[0]}")
            key = candidate_keys[0]
        else:
            raise KeyError(f"'{key}' not found in the state dictionary and no candidate keys were detected.")
    embedding_tensor = state_dict[key]
    # Build a mapping from word to embedding vector (as list)
    embeddings = {}
    for word, idx in vocab.items():
        embeddings[word] = embedding_tensor[idx].tolist()
    return embeddings

def main(vocab_path, cbow_model_path, output_path):
    # Load vocabulary
    vocab = load_vocab(vocab_path)
    print(f"Loaded vocabulary with {len(vocab)} words from {vocab_path}")

    # Load state dict from CBOW model
    cbow_state = torch.load(cbow_model_path, map_location=torch.device('cpu'))
    print(f"Loaded CBOW model state from {cbow_model_path}")

    # Extract embeddings
    embeddings = extract_embeddings(vocab, cbow_state)
    print(f"Extracted embeddings for {len(embeddings)} words")

    # Save the extracted embeddings to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)
    print(f"Saved extracted embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract word embeddings from a CBOW model state dictionary."
    )
    parser.add_argument(
        "--vocab", type=str, default="models/word2vec/text8_vocab.json",
        help="Path to the vocabulary JSON file."
    )
    parser.add_argument(
        "--model", type=str, default="models/word2vec/cbow_model_state.pth",
        help="Path to the CBOW model state (.pth) file."
    )
    parser.add_argument(
        "--output", type=str, default="models/word2vec/extracted_embeddings.json",
        help="Path to save the extracted embeddings."
    )
    args = parser.parse_args()
    
    # Normalize paths
    vocab_path = os.path.normpath(args.vocab)
    cbow_model_path = os.path.normpath(args.model)
    output_path = os.path.normpath(args.output)

    main(vocab_path, cbow_model_path, output_path)