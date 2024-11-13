import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,dims,n_layers):
        super().__init__()
        layers_list=[]
        for _ in range(n_layers):
            layers_list.append(nn.Linear(dims,dims//2))
            #layers_list.append(nn.BatchNorm1d(dims//2))
            layers_list.append(nn.ReLU())
            dims=dims//2
        layers_list.append(nn.Linear(dims,1))
        self.mlp=nn.Sequential(*layers_list)

    def forward(self,x):
        return self.mlp(x)
