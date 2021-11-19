#!/usr/bin/python
# -*- coding:utf-8 -*-
from datasets.CMAPSS_Graph_Datasets import datasets2
import numpy as np
import pandas as pd


class CMAPSS_graph(object):
    num_classes = 1
    feature = 30

    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file


    def data_preprare(self, test=False):
        if test:
            path = self.data_dir + 'test_' + self.data_file + '.pkl'
            test_pd = pd.read_pickle(path)
            edge_index = np.load(self.data_dir + 'edge_index_' + self.data_file + '.npy')
            edge_fe = np.load(self.data_dir + 'edge_feature_' + self.data_file + '.npy')
            test_dataset = datasets2(root=path, edge_index = edge_index)
            return test_dataset, test_pd
        else:
            train_path = self.data_dir + 'train_' + self.data_file + '.pkl'
            val_path = self.data_dir + 'test_' + self.data_file + '.pkl'
            edge_index = np.load(self.data_dir + 'edge_index_' + self.data_file + '.npy')

            train_dataset = datasets2(root=train_path, edge_index = edge_index)
            val_dataset = datasets2(root=val_path, edge_index = edge_index)
            return train_dataset, val_dataset




