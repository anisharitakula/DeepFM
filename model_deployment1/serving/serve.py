import mlflow.pytorch
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import torch
import uvicorn
import boto3
import os
import sys
from pathlib import Path
from typing import Tuple

# # Get the path to project root (parent of serving/)
# project_root = Path(__file__).parent.parent.absolute()

# # Add it to Python's search path
# sys.path.append(str(project_root))

# Add the root directory to Python path
sys.path.append('/app')


from config.config import S3_LOCATION
import s3fs
import json
from preprocess.collate_fn import fm_collate_fn



# Configure AWS credentials
def configure_aws_credentials():
    """Configure AWS credentials from environment variables"""
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key or not aws_secret_key:
        raise EnvironmentError(
            "AWS credentials not found. Please ensure AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables are set."
        )
    
    # Configure boto3 with credentials
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

class PredictionInput(BaseModel):
    data: Tuple[int,int,str]

app = FastAPI(title="Model Serving Endpoint")

# Load the model from MLflow
def load_model_from_uri():
    try:
        # Configure AWS credentials before loading model
        configure_aws_credentials()
        
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

        with open('model_uri.txt', 'r') as f:
            artifact_uri = f.read().strip()
        model_uri = f"{artifact_uri}/model"
        print(f"Attempting to load model from: {model_uri}")
        
        model = mlflow.pytorch.load_model(model_uri)
        return model, model_uri
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Initialize model with error handling
try:
    MODEL, model_uri = load_model_from_uri()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    MODEL = None
    model_uri = None

@app.get("/")
async def home():
    """
    Basic home page endpoint
    """
    return {
        "message": "Welcome to Model Serving Endpoint!",
        "model_uri": model_uri,
        "status": "operational" if MODEL is not None else "model loading failed"
    }

@app.post("/predict")
async def predict(request: PredictionInput):
    """
    Endpoint for model predictions
    """
    if MODEL is None:
        return {
            "error": "Model not properly initialized",
            "status": "error"
        }
    
    try:
        input = list(request.data)
        input[2] = input[2].split('|')
        
        
        # Initialize S3 filesystem
        fs = s3fs.S3FileSystem()

        
        with fs.open(f"{S3_LOCATION}/config_params/config_data.json", "r") as file:
            config_data = json.load(file)
        
    
        parsed_input=tuple(input)
        
        # Load the model
        model = mlflow.pytorch.load_model(model_uri)
        input_data=fm_collate_fn([parsed_input],**config_data,inference=True)
        model.eval()
        prediction=model(input_data)
        
        
        return {
            "predictions": prediction.tolist(),
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
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None
    }

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000)