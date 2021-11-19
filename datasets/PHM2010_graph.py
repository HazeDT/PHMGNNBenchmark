#!/usr/bin/python
# -*- coding:utf-8 -*-
from datasets.PHM2010_Graph_Datasets import datasets2
import numpy as np
import pandas as pd

class PHM2010_graph(object):
    num_classes = 1
    feature= 30


    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file

    def data_preprare(self, test=False):
        if test:
            name_list = self.data_file.split('_')  # the name of the data file must be 'c1_c4_c6'
            val_path = self.data_dir + name_list[2] + '.pkl'
            test_pd = pd.read_pickle(val_path)
            edge_index_3 = np.load(self.data_dir + 'edge_index_' + name_list[2] + '.npy')
            test_dataset = datasets2(root=val_path, edge_index=edge_index_3)
            return test_dataset, test_pd
        else:

            name_list = self.data_file.split('_')    #the name of the data file must be 'c1_c4_c6'
            train_path_1 = self.data_dir + name_list[0] + '.pkl'
            train_path_2 = self.data_dir + name_list[1] + '.pkl'
            val_path = self.data_dir + name_list[2] + '.pkl'

            edge_index_1 = np.load(self.data_dir + 'edge_index_' + name_list[0] + '.npy')
            edge_index_2 = np.load(self.data_dir + 'edge_index_' + name_list[1] + '.npy')
            edge_index_3 = np.load(self.data_dir + 'edge_index_' + name_list[2] + '.npy')

            train_dataset_1 = datasets2(root=train_path_1, edge_index = edge_index_1)
            train_dataset_2 = datasets2(root=train_path_2, edge_index = edge_index_2)
            train_dataset = train_dataset_1 + train_dataset_2
            val_dataset = datasets2(root=val_path,edge_index = edge_index_3)
            return train_dataset, val_dataset




