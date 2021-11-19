#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from torch_geometric.data import Data
import torch
from tqdm import tqdm


def datasets2(root, edge_index):

    anno_pd= pd.read_pickle(root)
    data_list = []
    features = anno_pd['data'].tolist()
    label = anno_pd['label'].tolist()
    cycle = anno_pd['cycle'].tolist()
    for i in tqdm(range(len(features))):
        x = features[i].T
        node_features = torch.tensor(x, dtype=torch.float)
        graph_label = torch.tensor([label[i]], dtype=torch.float)
        edge = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=node_features, y=graph_label, edge_index=edge)
        data_list.append(data)

    return data_list

