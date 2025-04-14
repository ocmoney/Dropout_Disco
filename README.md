# Dropout Disco 🕺💃

Predicting Hacker News Upvote Scores.

## Project Overview

This project aims to predict the upvote score of Hacker News posts based on their titles using deep learning techniques, specifically focusing on word embeddings and feature fusion.

## Directory Structure

```
├── app/              # Source code for the FastAPI application
│   ├── __init__.py   # Makes 'app' a Python package
│   └── main.py       # Main FastAPI application logic
├── notebooks/        # Jupyter notebooks for EDA, modeling, etc. (You might add this later)
├── src/              # Python source code for models, utils, etc. (You might add this later)
├── Dockerfile        # Defines the Docker image for the application
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd dropout-disco
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### Locally with Uvicorn

```bash
uvicorn app.main:app --reload
```
Access the API at `http://127.0.0.1:8000` and the docs at `http://127.0.0.1:8000/docs`.

### With Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t dropout-disco-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 --name dropout-disco-container dropout-disco-app
    ```
Access the API at `http://localhost:8000` (or your Docker host IP).

## TODO

*   Implement data loading and preprocessing.
*   Train word embedding model (Word2Vec).
*   Build and train regression model.
*   Integrate model into FastAPI prediction endpoint.
*   Add tests.
*   Refine Dockerfile for production.

