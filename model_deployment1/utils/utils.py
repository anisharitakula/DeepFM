import torch
import random
import numpy as np
import pandas as pd

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