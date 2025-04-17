#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OC_Collect_Test.py - Script for testing saved text-to-regression models
on Hacker News sentences.
"""

import os
import sys
import logging
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import project modules
from word2vec.vocabulary import Vocabulary
from utils import logger
from OC_Collect import TextToRegressionModel, predict_score, fetch_hacker_news_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.info("Starting OC_Collect_Test.py script")

# Database configuration
DB_URI = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(DB_URI)

def load_model(model_path, vocab_path, cbow_model_path, input_dim=128):
    """
    Load a saved model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        vocab_path (str): Path to the vocabulary file
        cbow_model_path (str): Path to the CBOW model file
        input_dim (int): Dimension of the input embeddings
        
    Returns:
        tuple: (loaded_model, log_transform)
    """
    # Create a new model instance
    model = TextToRegressionModel(
        vocab_path=vocab_path,
        cbow_model_path=cbow_model_path,
        input_dim=input_dim
    )
    
    # Load the saved model state
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get whether log transformation was used
    log_transform = checkpoint.get('log_transform', True)
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Log transformation: {'enabled' if log_transform else 'disabled'}")
    
    return model, log_transform

def find_saved_models(model_dir=None):
    """
    Find the most recent saved model in the specified directory.
    
    Args:
        model_dir (str, optional): Directory to search for models. If None, will search in both
                                  models/text_to_regression and models/text_to_regression_balanced.
    
    Returns:
        list: List containing the path to the most recent model file
    """
    # If no specific directory is provided, search in both directories
    if model_dir is None:
        # Try the original directory first
        original_dir = "models/text_to_regression"
        balanced_dir = "models/text_to_regression_balanced"
        
        # Check if either directory exists
        if os.path.exists(original_dir) and os.path.exists(balanced_dir):
            # Both directories exist, get the most recent model from either
            original_models = [f for f in os.listdir(original_dir) if f.startswith("model_") and f.endswith(".pth")]
            balanced_models = [f for f in os.listdir(balanced_dir) if f.startswith("model_") and f.endswith(".pth")]
            
            if not original_models and not balanced_models:
                logger.warning(f"No model files found in either {original_dir} or {balanced_dir}")
                return []
            
            # Get the most recent model from either directory
            all_models = []
            if original_models:
                all_models.extend([(os.path.join(original_dir, m), m) for m in original_models])
            if balanced_models:
                all_models.extend([(os.path.join(balanced_dir, m), m) for m in balanced_models])
            
            # Sort by timestamp in filename (newest first)
            all_models.sort(key=lambda x: x[1], reverse=True)
            most_recent_model = all_models[0][0]
            
            logger.info(f"Found most recent model: {os.path.basename(most_recent_model)} from {os.path.dirname(most_recent_model)}")
            return [most_recent_model]
        
        # Only one directory exists
        elif os.path.exists(original_dir):
            model_dir = original_dir
        elif os.path.exists(balanced_dir):
            model_dir = balanced_dir
        else:
            logger.error(f"Neither {original_dir} nor {balanced_dir} exists")
            return []
    
    # Use the specified directory
    if not os.path.exists(model_dir):
        logger.error(f"Directory {model_dir} does not exist")
        return []
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pth")]
    
    if not model_files:
        logger.warning(f"No model files found in {model_dir}")
        return []
    
    # Sort model files by timestamp in filename (newest first)
    model_files.sort(reverse=True)
    most_recent_model = model_files[0]
    model_path = os.path.join(model_dir, most_recent_model)
    
    logger.info(f"Found most recent model: {most_recent_model}")
    
    return [model_path]

def test_model_on_sentences(model, sentences, device, log_transform=True):
    """
    Test a model on a list of sentences.
    
    Args:
        model (TextToRegressionModel): Loaded model
        sentences (list): List of sentences to test
        device (torch.device): Device to run predictions on
        log_transform (bool): Whether the model was trained with log transformation
        
    Returns:
        dict: Dictionary mapping sentences to predicted scores
    """
    results = {}
    
    for sentence in sentences:
        predicted_score = predict_score(model, sentence, device, log_transform)
        results[sentence] = predicted_score
        
    return results

def analyze_predictions(results, actual_scores=None):
    """
    Analyze the predictions made by the model.
    
    Args:
        results (dict): Dictionary mapping sentences to predicted scores
        actual_scores (dict, optional): Dictionary mapping sentences to actual scores
        
    Returns:
        dict: Analysis results
    """
    predicted_scores = list(results.values())
    
    analysis = {
        "count": len(predicted_scores),
        "min": min(predicted_scores),
        "max": max(predicted_scores),
        "mean": np.mean(predicted_scores),
        "median": np.median(predicted_scores),
        "std": np.std(predicted_scores)
    }
    
    # Calculate percentiles with finer granularity in the tail
    # Regular percentiles from 0 to 95 in steps of 5
    regular_percentiles = list(range(0, 96, 5))
    # Fine-grained percentiles from 95 to 100 in steps of 1
    tail_percentiles = list(range(96, 101))
    # Combine both lists
    percentiles = regular_percentiles + tail_percentiles
    
    for p in percentiles:
        analysis[f"p{p}"] = np.percentile(predicted_scores, p)
    
    # If actual scores are provided, calculate error metrics
    if actual_scores:
        errors = []
        for sentence, predicted in results.items():
            if sentence in actual_scores:
                actual = actual_scores[sentence]
                error = abs(predicted - actual)
                errors.append(error)
        
        if errors:
            analysis["mae"] = np.mean(errors)
            analysis["rmse"] = np.sqrt(np.mean(np.square(errors)))
            analysis["median_ae"] = np.median(errors)
    
    return analysis

def plot_score_distribution(results, title="Predicted Score Distribution"):
    """
    Plot the distribution of predicted scores.
    
    Args:
        results (dict): Dictionary mapping sentences to predicted scores
        title (str): Title for the plot
    """
    predicted_scores = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_scores, bins=30, alpha=0.7, color='blue')
    plt.axvline(np.mean(predicted_scores), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(predicted_scores):.2f}')
    plt.axvline(np.median(predicted_scores), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(predicted_scores):.2f}')
    plt.title(title)
    plt.xlabel('Predicted Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    save_dir = "models/text_to_regression"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "score_distribution.png"))
    plt.close()
    
    logger.info(f"Score distribution plot saved to {os.path.join(save_dir, 'score_distribution.png')}")

def get_sample_sentences(data, num_samples=100):
    """
    Get a sample of sentences from the Hacker News dataset.
    
    Args:
        data (pandas.DataFrame): Hacker News dataset
        num_samples (int): Number of samples to get
        
    Returns:
        list: List of sample sentences
    """
    # Get a random sample of titles
    sample_titles = data['title'].sample(min(num_samples, len(data))).tolist()
    
    # Also get some high-scoring titles
    high_scoring = data.nlargest(10, 'score')['title'].tolist()
    
    # Also get some low-scoring titles
    low_scoring = data.nsmallest(10, 'score')['title'].tolist()
    
    # Combine all samples
    all_samples = sample_titles + high_scoring + low_scoring
    
    # Remove duplicates
    unique_samples = list(set(all_samples))
    
    logger.info(f"Selected {len(unique_samples)} unique sentences for testing")
    
    return unique_samples

def get_test_dataset():
    """
    Get the entire test dataset used for model evaluation.
    
    Returns:
        tuple: (test_sentences, test_scores)
    """
    # Fetch Hacker News data
    data = fetch_hacker_news_data()
    
    # Extract titles and scores
    titles_and_scores = data.loc[:, ['title', 'score']].copy()
    
    # Split the data the same way as in OC_Collect.py
    _, test_df = train_test_split(titles_and_scores, test_size=0.2, random_state=42)
    
    # Get all test sentences and their scores
    test_sentences = test_df['title'].tolist()
    test_scores = test_df['score'].tolist()
    
    logger.info(f"Retrieved entire test dataset with {len(test_sentences)} sentences")
    
    return test_sentences, test_scores

def main():
    """Main function to run the script."""
    # Find the most recent saved model
    model_paths = find_saved_models()
    
    if not model_paths:
        logger.error("No models found. Please run OC_Collect.py or OC_Collect_EvenScore.py first to train and save a model.")
        return
    
    # Get the entire test dataset
    test_sentences, test_scores = get_test_dataset()
    
    # Create a dictionary mapping sentences to their actual scores
    actual_scores = {sentence: score for sentence, score in zip(test_sentences, test_scores)}
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Test the most recent model
    model_path = model_paths[0]
    logger.info(f"\nTesting most recent model: {model_path}")
    
    # Load the model
    model, log_transform = load_model(
        model_path=model_path,
        vocab_path="models/word2vec/text8_vocab_NW10M_MF5.json",
        cbow_model_path="models/word2vec/CBOW_D128_W3_NW10M_MF5_E5_LR0.001_BS512/model_state.pth",
        input_dim=128
    )
    
    # Move model to device
    model = model.to(device)
    
    # Test the model on the entire test dataset
    results = test_model_on_sentences(model, test_sentences, device, log_transform)
    
    # Analyze the predictions
    analysis = analyze_predictions(results, actual_scores)
    
    # Print analysis
    logger.info("\nPrediction Analysis:")
    logger.info(f"  Count: {analysis['count']}")
    logger.info(f"  Min: {analysis['min']:.2f}")
    logger.info(f"  Max: {analysis['max']:.2f}")
    logger.info(f"  Mean: {analysis['mean']:.2f}")
    logger.info(f"  Median: {analysis['median']:.2f}")
    logger.info(f"  Std: {analysis['std']:.2f}")
    
    # Print percentiles in a table format
    logger.info("\nPercentile Distribution:")
    logger.info("Percentile | Predicted Score")
    logger.info("-" * 30)
    
    # Regular percentiles from 0 to 95 in steps of 5
    for p in range(0, 96, 5):
        logger.info(f"{p:3d}th     | {analysis[f'p{p}']:.2f}")
    
    # Fine-grained percentiles from 95 to 100 in steps of 1
    for p in range(96, 101):
        logger.info(f"{p:3d}th     | {analysis[f'p{p}']:.2f}")
    
    if 'mae' in analysis:
        logger.info("\nError Metrics:")
        logger.info(f"  MAE: {analysis['mae']:.2f}")
        logger.info(f"  RMSE: {analysis['rmse']:.2f}")
        logger.info(f"  Median AE: {analysis['median_ae']:.2f}")
    
    # Plot score distribution
    plot_score_distribution(results, title=f"Predicted Score Distribution - {os.path.basename(model_path)}")
    
    # Print some example predictions (first 10)
    logger.info("\nExample Predictions (first 10):")
    for i, (sentence, score) in enumerate(list(results.items())[:10]):
        actual = actual_scores.get(sentence, "N/A")
        logger.info(f"{i+1}. '{sentence}' -> {score:.2f} (Actual: {actual})")
    
    # Save results to file
    save_dir = os.path.dirname(model_path)
    results_file = os.path.join(save_dir, f"results_{os.path.basename(model_path).replace('.pth', '.json')}")
    
    with open(results_file, "w") as f:
        json.dump({
            "model": model_path,
            "log_transform": log_transform,
            "results": results,
            "analysis": analysis
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()

