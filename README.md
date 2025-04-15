# Hacker News Upvote Prediction 🚀

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Dropout Disco** (Yurii, James, Ollie, Emil) for Week 1 of the MLX Institute Intensive Program.

## Project Overview

This project aims to predict the upvote score of Hacker News posts. The initial phase focuses on generating high-quality word embeddings using a custom Word2Vec (CBOW) model trained on the `text8` corpus. These embeddings will later be used as features for a regression model to predict the scores based on post titles.

**Phase 1 (Word Embeddings) is complete!** ✅

## Key Features Implemented (Phase 1)

*   **Custom Word2Vec (CBOW):** Implementation using PyTorch (`src/word2vec/model.py`).
*   **Vocabulary Management:** Handles creation, filtering (`min_freq`, `<UNK>`), saving, and loading (`src/word2vec/vocabulary.py`, `models/word2vec/`).
*   **Dataset Preparation:** Generates context-target pairs for CBOW (`src/word2vec/dataset.py`).
*   **Configurable Training:** Uses `config.yaml` for hyperparameters and paths, with command-line overrides via `argparse` (`train_word2vec.py`).
*   **Modular Training Loop:** Organized training logic (`src/word2vec/trainer.py`).
*   **MPS Acceleration:** Utilizes Apple Silicon GPU acceleration via MPS if available (`utils/device_setup.py`).
*   **Experiment Tracking (W&B):** Integrated Weights & Biases for logging metrics, configuration, and artifacts (`train_word2vec.py`).
*   **Organized Artifacts:** Saves model state, vocabulary, and loss plots/data into run-specific directories based on hyperparameters (`models/word2vec/RUN_NAME/...`).
*   **Logging:** Centralized logging setup (`utils/logging.py`).
*   **Evaluation:** Basic intrinsic evaluation notebook for checking embedding quality (`notebooks/03_evaluate_word2vec.ipynb`).

## Directory Structure

```
.
├── app/              # Placeholder for future FastAPI app
├── config.yaml       # Central configuration for paths & hyperparameters
├── data/             # Input data (e.g., text8.txt)
├── Dockerfile        # Placeholder for future deployment
├── docs/             # Project documentation (dev plan, etc.)
├── logs/             # Log files generated during runs
├── models/           # Saved model artifacts
│   ├── regression/   # Placeholder for regression model
│   └── word2vec/     # Vocabulary files & run-specific model dirs
│       ├── text8_vocab_*.json
│       └── CBOW_D128_W3_NW10M_MF5_E5_LR0.001_BS512/ # Example run dir
│           ├── model_state.pth
│           ├── cbow_training_losses.json
│           └── cbow_training_loss.png
├── notebooks/        # Jupyter notebooks (EDA, evaluation)
├── README.md         # This file
├── requirements.txt  # Project dependencies
├── src/              # Core Python source code
│   ├── __init__.py
│   ├── regression/   # Placeholder for regression code
│   └── word2vec/     # Word2Vec implementation modules
│       ├── __init__.py
│       ├── dataset.py
│       ├── model.py
│       ├── trainer.py
│       └── vocabulary.py
├── train_word2vec.py # Main script to train Word2Vec model  <-- ENTRY POINT
├── utils/            # Utility modules (logging, device setup)
│   ├── __init__.py
│   ├── device_setup.py
│   └── logging.py
└── wandb/            # Local W&B logs/cache (in .gitignore)
```

*(Note: The `scripts/` directory seen previously seems unused; `train_word2vec.py` in the root is the current entry point.)*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd dropout-disco # Or your repo name
    ```

2.  **Create & Activate Environment:** (Using Conda or venv)
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate

    # Using Conda
    # conda create -n dropout-disco python=3.11 # Or your desired version
    # conda activate dropout-disco
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Weights & Biases Login (First time):**
    ```bash
    wandb login
    ```
    (Follow prompts to paste your API key)

5.  **Download Data:** Ensure the `text8.txt` corpus file is placed in the `data/` directory.

## Running Word2Vec Training

The primary script for training the Word2Vec model is `train_word2vec.py`.

1.  **Configure:** Review and modify `config.yaml` to set desired hyperparameters (embedding dimension, window size, epochs, learning rate, etc.) and paths.

2.  **Run Training:** Execute the script from the project root directory.
    ```bash
    python train_word2vec.py
    ```

3.  **Override Parameters (Optional):** You can override settings from `config.yaml` using command-line arguments. For example, to run a quick test:
    ```bash
    python train_word2vec.py --num-words 100000 --epochs 1 --batch-size 64 --no-wandb
    ```
    Use `python train_word2vec.py --help` to see all available arguments.

4.  **Monitor (W&B):** If W&B logging is enabled (default), the script will output a link to the W&B run page where you can monitor training progress (loss curves) and view logged artifacts.

## Next Steps (Phase 2)

*   ✅ Implement data loading for Hacker News posts (titles and scores).
*   🏗️ Build a regression model using PyTorch (`src/regression/`).
*   🧠 Load the pre-trained Word2Vec embeddings (`models/word2vec/`).
*   🧩 Implement feature extraction: Convert HN titles into fixed-size vectors using the loaded embeddings (e.g., averaging).
*   🚂 Train the regression model to predict scores based on the title embeddings.
*   📊 Evaluate regression model performance.
*   ☁️ Integrate the trained regression model into the FastAPI prediction endpoint (`app/`).
*   🐳 Refine Dockerfile for the prediction service.
*   🧪 Add tests.
