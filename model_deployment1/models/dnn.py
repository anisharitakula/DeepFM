import torch.nn as nn
from models.mlp import MLP
import torch

class DNN(nn.Module):
    def __init__(self,factors,dims,n_layers=3):
        super(DNN,self).__init__()
        self.factors=factors
        self.mlp=MLP(dims,n_layers)
    
    def forward(self,input_data):
        batch_indices,feature_indices=input_data.coalesce().indices()
        embeds=self.factors(feature_indices)
        batch_embeds=[]
        for i in range(batch_indices.max()+1):
            batch_mask=(batch_indices==i)
            batch_embeds.append(torch.mean(embeds[batch_mask],dim=0))
        
        x=torch.stack(batch_embeds)
        x=self.mlp(x)
    
        return x
