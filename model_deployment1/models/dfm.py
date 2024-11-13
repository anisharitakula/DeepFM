import torch.nn as nn
import torch
from models.dnn import DNN


class ModelDFM(nn.Module):
    def __init__(self,embed_dim,n_dims):
        super().__init__()
        self.embed_dim=embed_dim    
        self.weights=nn.Linear(n_dims,1)
        self.factors=nn.Embedding(n_dims,embed_dim)
        self.dnn=DNN(self.factors,embed_dim)

        for embedding in [self.weights,self.factors]:
            nn.init.xavier_uniform_(embedding.weight)

        
    def forward(self,input_data):
        x_reg=torch.sparse.mm(input_data,self.weights.weight.t()) + self.weights.bias
        sum_squared=torch.square(torch.matmul(input_data,self.factors.weight).sum(dim=1))
        squared_sum=torch.matmul(torch.square(input_data),torch.square(self.factors.weight)).sum(dim=1)
        x=x_reg + .5*(sum_squared-squared_sum).view(-1,1) + self.dnn(input_data)
        x=0.5 + torch.sigmoid(x)*5
        return x.squeeze()