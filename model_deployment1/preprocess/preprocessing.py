import pandas as pd
import numpy as np


def process_data(dataset1_s3loc,dataset2_s3loc):

    

    # Retrieve the file from S3
    data=pd.read_csv(dataset1_s3loc)
    movie_data=pd.read_csv(dataset2_s3loc)

    movie_genres=set()
    for genre in movie_data['genres']:
        movie_genres.update(genre.split('|'))
    movie_genres_dict=dict(zip(movie_genres,range(len(movie_genres))))

    movie_data['genres']=movie_data['genres'].map(lambda x:x.split('|'))
    data_tags=pd.merge(data,movie_data,how="left",on="movieId")

    
    unique_users=data_tags['userId'].nunique()
    unique_movies=data_tags['movieId'].nunique()
    n_records=data_tags.shape[0]
    
    user_id_map=dict(zip(data_tags['userId'].unique(),range(unique_users)))
    movie_id_map=dict(zip(data_tags['movieId'].unique(),range(unique_movies)))

    list_indices=np.random.choice(n_records,int(n_records*.7),replace=False)
    
    #Mapping user and movie
    data_tags['user_index']=data_tags['userId'].map(user_id_map)
    data_tags['movie_index']=data_tags['movieId'].map(movie_id_map)

    train_data,test_data=data_tags.iloc[list_indices],data_tags.iloc[[False if i in list_indices else True for i in range(n_records)]]
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)
    return train_data,test_data
