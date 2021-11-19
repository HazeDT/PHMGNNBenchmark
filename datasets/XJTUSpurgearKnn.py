import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.KNNGraph import KNNGraph
from datasets.AuxFunction import FFT
from tqdm import tqdm
import pickle
# -------------------------------------------------------------
signal_size = 1024

# label
label = [i for i in range(10)]


# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    file = ['15Hz','20Hz']

    Subdir = []
    for i in file:
        sub_root = os.path.join('/tmp', root, i)
        file_name = os.listdir(sub_root)
        for j in file_name:
            Subdir.append(os.path.join('/tmp', sub_root, j))

    data = []

    for i in tqdm(range(len(Subdir))):

        data1 = data_load(sample_length, Subdir[i],label=label[i],InputType=InputType,task=task)
        data += data1

    return data


def data_load(signal_size, root, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    root:Data location
    '''

    fl = pd.read_csv(root,sep='\t',usecols=[1],header=None,)
    fl = fl.values
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1,)
    data = []
    start, end = 0, signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        if InputType == "TD":
            x = fl[start:end]
        elif InputType == "FD":
            x = fl[start:end]
            x = FFT(x)
        else:
            print("The InputType is wrong!!")

        data.append(x)
        start += signal_size
        end += signal_size

    graphset = KNNGraph(10,data,label,task)

    return graphset


class XJTUSpurgearKnn(object):
    num_classes = 10

    def __init__(self, sample_length, data_dir,InputType,task):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task



    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            with open(os.path.join(self.data_dir, "XJTUSpurgearKnn.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset

