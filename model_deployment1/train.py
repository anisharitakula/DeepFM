import mlflow
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
import typer
import torch
from utils.utils import set_seed, get_unique_movies, get_unique_users, get_movie_genres
from preprocess.preprocessing import process_data
from preprocess.sparse_dataset import SparseMatrixDataset
from preprocess.collate_fn import fm_collate_fn
from models.dfm import ModelDFM
from models.base_learner import BaseLearner
from config.config import EXPERIMENT_NAME, SEED
from typing_extensions import Annotated
import os

app = typer.Typer()

@app.command()
def train(
    dataset1_s3loc: Annotated[str, typer.Option(help="ratings dataset")] = None,
    dataset2_s3loc: Annotated[str, typer.Option(help="movies dataset")] = None,
    embed_dim: Annotated[int, typer.Option(help="dimensionality of embeddings")] = 5,
    lr: Annotated[float, typer.Option(help="learning rate for model")] = 0.001,
    epochs: Annotated[int, typer.Option(help="Number of epochs")] = 4,
):
    # Ensure no lingering MLflow runs
    # if mlflow.active_run():
    #     print(f"Ending lingering active run: {mlflow.active_run().info.run_id}")
    #     mlflow.end_run()

    start_time = time.time()
    set_seed(SEED)

    #print(f"Active artifact URI: {mlflow.get_artifact_uri()}")

    # Create or get the experiment ID
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME,artifact_location=os.environ['MLFLOW_ARTIFACT_URI'])
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    # Load and process datasets
    data = pd.read_csv(dataset1_s3loc)
    movie_data = pd.read_csv(dataset2_s3loc)
    train_data, test_data = process_data(data, movie_data)

    unique_users = get_unique_users(data)
    unique_movies = get_unique_movies(data)
    movie_genres_dict = get_movie_genres(movie_data)

    train_dataset = SparseMatrixDataset(train_data)
    test_dataset = SparseMatrixDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: fm_collate_fn(
            batch, unique_users, unique_movies, movie_genres_dict
        ),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: fm_collate_fn(
            batch, unique_users, unique_movies, movie_genres_dict
        ),
    )

    model = ModelDFM(embed_dim, unique_users + unique_movies + len(movie_genres_dict))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    cb = BaseLearner(train_dataloader, test_dataloader, model, optimizer, lr, criterion)
    val_loss = cb.fit(epochs)

    baseline = [np.mean(test_data["rating"])] * len(test_data["rating"])
    mean_square_error = ((test_data["rating"] - baseline) ** 2).mean()
    print(f"The val loss for baseline prediction is {mean_square_error}")
    end_time = time.time()
    print(f"Total execution time in minutes is {(end_time - start_time) / 60}")

    # Log the run using MLflow context manager
    with mlflow.start_run(experiment_id=experiment_id):

        #Get mlflow tracking uri
        print(f"Mlflow tracking uri: {mlflow.get_tracking_uri()}")

        # Log parameters
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("random_seed", SEED)
        mlflow.log_param("epochs", epochs)

        # Log a metric
        mlflow.log_metric("final_loss", val_loss)

        # Getting the artifact uri
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

        # Log the PyTorch model
        mlflow.pytorch.log_model(model,artifact_path='model')

        

        print("Model logged successfully!")

if __name__ == "__main__":
    app()
