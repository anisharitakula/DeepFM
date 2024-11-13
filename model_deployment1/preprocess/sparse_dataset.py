from torch.utils.data import Dataset

class SparseMatrixDataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        user_idx=self.data.iloc[idx]['user_index']
        movie_idx=self.data.iloc[idx]['movie_index']
        genres=self.data.iloc[idx]['genres']
        rating=self.data.iloc[idx]['rating']
        
        return user_idx,movie_idx,genres,rating
