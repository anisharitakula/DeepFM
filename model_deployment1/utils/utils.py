import torch
import random
import numpy as np
import pandas as pd
import mlflow
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_unique_users(data):
    return data['userId'].nunique()

def get_unique_movies(data):
    return data['movieId'].nunique()

def get_movie_genres(movie_data):
    movie_genres=set()
    for genre in movie_data['genres']:
        movie_genres.update(genre.split('|'))
    movie_genres_dict=dict(zip(movie_genres,range(len(movie_genres))))
    return movie_genres_dict

def get_best_run_id(experiment_name: str, metric: str = "val_loss"):
    # Set tracking URI
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # Get experiment details
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    pd.set_option('display.max_columns', None)
    
    # Search for all runs in the experiment
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} ASC"],  # Sort by validation loss (smallest first)
        max_results=1                       # Fetch only the top result
    )
    
    if runs_df.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}' with metric '{metric}'.")
    
    # Extract the best run ID
    best_run_id = runs_df.loc[0, "run_id"]
    artifact_uri = runs_df.loc[0, "artifact_uri"]
    print(runs_df)
    print(f"Best run ID: {best_run_id}")
    print(f"Artifact uri is: {artifact_uri}")
    return best_run_id,artifact_uri
