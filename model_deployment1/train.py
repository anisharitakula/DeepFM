#Factorization Machines(MSE val loss -.82)

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
import typer
import torch
import json

from utils.utils import set_seed,get_unique_movies,get_unique_users,get_movie_genres
from preprocess.preprocessing import process_data
from preprocess.sparse_dataset import SparseMatrixDataset
from preprocess.collate_fn import fm_collate_fn
from models.dfm import ModelDFM
from models.base_learner import BaseLearner
from typing_extensions import Annotated

pd.set_option('display.max_rows', None)

#Initialize the typer app
app=typer.Typer()

@app.command()
def train(dataset1_loc: Annotated[str, typer.Option(help="ratings dataset")] = None,
          dataset2_loc: Annotated[str, typer.Option(help="movies dataset")] = None,
          embed_dim: Annotated[int, typer.Option(help="dimensionality of embeddings")]=5,
          lr: Annotated[float, typer.Option(help="learning rate for model")]=.001,
          epochs: Annotated[int, typer.Option(help="Number of epochs")]=4):

    start_time=time.time()
    set_seed(82)
    
    train_data,test_data=process_data(dataset1_loc,dataset2_loc)
    
    unique_users=get_unique_users(dataset1_loc)
    unique_movies=get_unique_movies(dataset1_loc)
    movie_genres_dict=get_movie_genres(dataset2_loc)
    
    config_dict={'unique_users':unique_users,'unique_movies':unique_movies,'movie_genres_dict':movie_genres_dict}
    with open("config/config_data.json", "w") as json_file:
        json.dump(config_dict, json_file)

    with open("config/model_params.json","w") as file:
        json.dump({"embed_dim":embed_dim,"lr":lr,"epochs":epochs},file)

    train_dataset=SparseMatrixDataset(train_data)
    test_dataset=SparseMatrixDataset(test_data)
   
    train_dataloader= DataLoader(train_dataset,batch_size=8,shuffle=True,collate_fn=lambda batch: fm_collate_fn(batch, unique_users,unique_movies,movie_genres_dict)
)
    test_dataloader= DataLoader(test_dataset,batch_size=8,shuffle=True,collate_fn=lambda batch: fm_collate_fn(batch, unique_users,unique_movies,movie_genres_dict)
)

    model = ModelDFM(embed_dim,unique_users+unique_movies+len(movie_genres_dict))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    criterion=nn.MSELoss()

    cb=BaseLearner(train_dataloader,test_dataloader,model,optimizer,lr,criterion)
    cb.fit(epochs)

    baseline=[np.mean(test_data['rating'])]*len(test_data['rating'])
    mean_square_error=((test_data['rating']-baseline)**2).mean()
    print(f"The val loss for baseline prediction is {mean_square_error}")
    end_time=time.time()
    print(f"Total execution time in minutes is {(end_time-start_time)/60}")

    #Persisting the model
    torch.save(model.state_dict(), "saved_models/deepfm_model.pth")


if __name__=="__main__":
    app()   


