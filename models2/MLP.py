import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, feature, out_channel, pooltype):
        super(MLP, self).__init__()

        self.pool1, self.pool2 = self.poollayer(pooltype)

        self.fc1 = nn.Sequential(
            nn.Linear(feature, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Sequential(nn.Linear(512, out_channel))


    def forward(self, data, pooltype):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.fc1(x)
        x, edge_index, batch = self.poolresult(self.pool1,pooltype,x, edge_index, batch)
        x1 = global_mean_pool(x, batch)
        x = self.fc2(x)
        x, edge_index, batch = self.poolresult(self.pool2, pooltype, x, edge_index, batch)
        x2 = global_mean_pool(x, batch)
        x = x1 + x2
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return x

    def poollayer(self, pooltype):

        self.pooltype = pooltype

        if self.pooltype == 'TopKPool':
            self.pool1 = TopKPooling(1024)
            self.pool2 = TopKPooling(1024)
        elif self.pooltype == 'EdgePool':
            self.pool1 = EdgePooling(1024)
            self.pool2 = EdgePooling(1024)
        elif self.pooltype == 'ASAPool':
            self.pool1 = ASAPooling(1024)
            self.pool2 = ASAPooling(1024)
        elif self.pooltype == 'SAGPool':
            self.pool1 = SAGPooling(1024)
            self.pool2 = SAGPooling(1024)
        else:
            print('Such graph pool method is not implemented!!')

        return self.pool1, self.pool2

    def poolresult(self,pool,pooltype,x,edge_index,batch):

        self.pool = pool

        if pooltype == 'TopKPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'EdgePool':
            x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'ASAPool':
            x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'SAGPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        else:
            print('Such graph pool method is not implemented!!')

        return x, edge_index, batch