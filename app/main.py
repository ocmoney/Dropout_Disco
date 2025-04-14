from fastapi import FastAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dropout Disco API", version="0.1.0")

@app.on_event("startup")
async def startup_event():
    logger.info("Dropout Disco API starting up...")

@app.get("/")
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Dropout Disco API!"}

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

# Example prediction endpoint placeholder
# @app.post("/predict")
# async def predict_score(title: str):
#     # TODO: Implement prediction logic using your trained model
#     logger.info(f"Prediction requested for title: {title}")
#     # score = model.predict(title) # Placeholder
#     score = 42 # Dummy score
#     return {"title": title, "predicted_score": score}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Dropout Disco API shutting down...")

