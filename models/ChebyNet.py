import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj

class ChebyNet(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(ChebyNet, self).__init__()

        self.GConv1 = ChebConv(feature,1024,K=1)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = ChebConv(1024,1024,K=1)
        self.bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.GConv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        edge_index, edge_weight = dropout_adj(edge_index, edge_weight)

        x = self.GConv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x