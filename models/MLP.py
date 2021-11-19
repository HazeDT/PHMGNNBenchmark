import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa


class MLP(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(MLP, self).__init__()

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


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return x