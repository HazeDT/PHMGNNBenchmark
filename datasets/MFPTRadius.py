import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.RadiusGraph import RadiusGraph
from datasets.AuxFunction import FFT
import pickle

#-------------------------------------------------------------
signal_size=1024


#label
label1 = [1,2,3,4,5,6,7]
label2 = [8,9,10,11,12,13,14]   #The failure data is labeled 1-14

#generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    m = os.listdir(root)
    datasetname = os.listdir(os.path.join("/tmp", root, m[0]))  # '1 - Three Baseline Conditions'
    # '2 - Three Outer Race Fault Conditions'
    # '3 - Seven More Outer Race Fault Conditions'
    # '4 - Seven Inner Race Fault Conditions'
    # '5 - Analyses',
    # '6 - Real World Examples
    # Generate a list of data
    dataset1 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[0]))  # 'Three Baseline Conditions'
    dataset2 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[2]))  # 'Seven More Outer Race Fault Conditions'
    dataset3 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[3]))  # 'Seven Inner Race Fault Conditions'
    data_root1 = os.path.join('/tmp', root, m[0], datasetname[0])  # Path of Three Baseline Conditions
    data_root2 = os.path.join('/tmp', root, m[0], datasetname[2])  # Path of Seven More Outer Race Fault Conditions
    data_root3 = os.path.join('/tmp', root, m[0], datasetname[3])  # Path of Seven Inner Race Fault Conditions

    path1 = os.path.join('/tmp', data_root1, dataset1[0])
    data = data_load(sample_length, path1, label=0, InputType=InputType, task=task)  # The label for normal data is 0

    for i in tqdm(range(len(dataset2))):
        path2 = os.path.join('/tmp', data_root2, dataset2[i])
        data1 = data_load(sample_length, path2, label=label1[i], InputType=InputType, task=task)
        data += data1

    for j in tqdm(range(len(dataset3))):
        path3 = os.path.join('/tmp', data_root3, dataset3[j])
        data2 = data_load(sample_length, path3, label=label2[j], InputType=InputType, task=task)
        data += data2

    return data


def data_load(signal_size, filename, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    if label == 0:
        fl = (loadmat(filename)["bearing"][0][0][1])  # Take out the data
    else:
        fl = (loadmat(filename)["bearing"][0][0][2])  # Take out the data
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1, )
    data = []
    start, end = 0, signal_size
    while end <= fl[:signal_size * 1000].shape[0]:
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

    graphset = RadiusGraph(10,data,label,task)

    return graphset


#--------------------------------------------------------------------------------------------------------------------
class MFPTRadius(object):
    num_classes = 15

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
            with open(os.path.join(self.data_dir, "MFPTRadius.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset

