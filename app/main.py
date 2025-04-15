# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: app/main.py
# Description: FastAPI application instance and API endpoints.
# Created: 2025-04-15
# Updated: 2025-04-15

import os
from fastapi import FastAPI
# Add other necessary imports later

# TODO: Define model_version based on saved model
model_version = "0.0.0" 

log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"

# Ensure log directory exists (useful for local runs, Docker handles volume)
# os.makedirs(log_dir_path, exist_ok=True) 

app = FastAPI()

# TODO: Import predictor functions
# from .predictor import load_models, predict # Example

# TODO: Load models on startup (implement robustly)
# load_models() 

@app.get("/ping")
def ping():
    return "ok"

@app.get("/version")
def version():
    # TODO: Get version dynamically if possible
    return {"version": model_version}

# TODO: Implement logging functions and endpoint
@app.get("/logs")
def logs():
    # return read_logs(log_path)
    return {"logs": ["Log reading not implemented yet."]}

# TODO: Define input model (Pydantic)
# class PostInput(BaseModel):
#     author: str | None = None # Allow optional fields if needed
#     title: str
#     timestamp: str | None = None # Or datetime

@app.post("/how_many_upvotes")
# TODO: Replace 'post: dict' with Pydantic model: post: PostInput
def how_many_upvotes(post: dict): 
    # start_time = os.times().elapsed
    # prediction = predict(post.title) # Call predictor function
    # end_time = os.times().elapsed
    # latency = (end_time - start_time) * 1000

    # message = {
    #    "Latency": latency,
    #    "Version": model_version,
    #    "Timestamp": end_time, # Or use datetime.now()
    #    "Input": post, # Or post.dict() if using Pydantic
    #    "Prediction": prediction
    # }
    # log_request(log_path, message) # Implement logging

    prediction = 0 # Placeholder
    return {"upvotes": prediction}

# Placeholder for logging functions - Implement these
def log_request(path, msg):
    print(f"LOG (to {path}): {msg}") # Basic print logging for now
    # with open(path, 'a') as f:
    #     f.write(json.dumps(msg) + '\n')
    pass

def read_logs(path):
     # Implement log reading
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-50:]]} # Return last 50 lines
    except FileNotFoundError:
        return {"logs": ["Log file not found."]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}


