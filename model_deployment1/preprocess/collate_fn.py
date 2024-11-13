import torch

def fm_collate_fn(batch,unique_users,unique_movies,movie_genres_dict,inference=False):
    user_movie_indices=[]
    ratings=[]
    row_indices=[]

    i=0
    

    for data in batch:
        if inference:
            user_idx,movie_idx,genre=data
        else:
            user_idx,movie_idx,genre,rating=data
        
        a=[int(user_idx),int(movie_idx+unique_users)]
        b=[unique_users+unique_movies+movie_genres_dict[key] for key in genre]
        
        user_movie_indices.extend(a+b)
        row_indices.extend([i,i] + [i]*len(b))

        if not inference:
            ratings.append(torch.tensor(rating,dtype=torch.float))
        i+=1
    
    ratings = torch.stack(ratings) if ratings else None
    input=torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices,user_movie_indices]),
        values=torch.ones(len(row_indices)),
        size=(i,unique_users+unique_movies+len(movie_genres_dict))
    )
    return (input,ratings) if not inference else input

