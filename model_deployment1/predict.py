import typer
from models.dfm import ModelDFM
import torch
import json
from preprocess.collate_fn import fm_collate_fn
from typing_extensions import Annotated
import ast
import mlflow
import os
import s3fs
from utils.utils import get_best_run_id
from config.config import S3_LOCATION,EXPERIMENT_NAME

app=typer.Typer()

@app.command()
def predict(input: Annotated[str,typer.Option(help="tuple of user_id,movie_id and genre details") ]):
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    #Get run_id and artifact uri
    run_id,artifact_uri=get_best_run_id(EXPERIMENT_NAME)

    model_uri=f"{artifact_uri}/model"
    print(f"The model_uri is {model_uri}")

    # Construct the model URI
    #model_uri = f"runs:/{run_id}/model"

    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem()

    parsed_input=list(ast.literal_eval(input))
    with fs.open(f"{S3_LOCATION}/config_params/config_data.json", "r") as file:
        config_data = json.load(file)
    
    
    parsed_input[2]=parsed_input[2].split('|')
    parsed_input=tuple(parsed_input)
    
    # Load the model
    model = mlflow.pytorch.load_model(model_uri)
    input_data=fm_collate_fn([parsed_input],**config_data,inference=True)
    model.eval()
    pred=model(input_data)
    print(float(pred))
    return float(pred)

if __name__=="__main__":
    app()
