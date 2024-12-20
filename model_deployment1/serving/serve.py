import mlflow.pytorch
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import torch

class PredictionInput(BaseModel):
    data: list[int,int,str]

app = FastAPI(title="Model Serving Endpoint")

# Load the model from MLflow
def load_model_from_uri():
    with open('model_uri.txt', 'r') as f:
        artifact_uri = f.read().strip()
    model_uri=f"{artifact_uri}/model"
    return mlflow.pytorch.load_model(model_uri),model_uri

# Global model variable
MODEL,model_uri = load_model_from_uri()

@app.get("/")
async def home():
    """
    Basic home page endpoint
    """
    return {
        "message": "Welcome to Model Serving Endpoint!",
        "model_uri": model_uri,
        "status": "operational"
    }

@app.post("/predict")
async def predict(request: PredictionInput):
    """
    Endpoint for model predictions
    """
    try:
        input=request.data
        input[2]=input[2].split('|')
        input_data=tuple(input)
        
        # Generate predictions
        predictions = MODEL.predict(input_data)
        
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}


