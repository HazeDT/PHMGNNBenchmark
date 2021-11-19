import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.RadiusGraph import RadiusGraph
from datasets.AuxFunction import FFT
import pickle
# ------------------------------------------------------------
signal_size = 1024
root = "D:\data\XJTU_Gearbox"

fault_name = ['1ndBearing_ball','1ndBearing_inner','1ndBearing_mix(inner+outer+ball)','1ndBearing_outer',
              '2ndPlanetary_brokentooth','2ndPlanetary_missingtooth','2ndPlanetary_normalstate','2ndPlanetary_rootcracks','2ndPlanetary_toothwear']
# label
label = [i for i in range(9)]



# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task,test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data = []

    for i in tqdm(range(len(fault_name))):
        data_name = 'Data_Chan1.txt'
        path2 = os.path.join('/tmp', root, fault_name[i],data_name)
        data1 = data_load(sample_length,filename=path2, label=label[i],InputType=InputType,task=task)
        data += data1

    return data


def data_load(signal_size,filename, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = pd.read_csv(filename, skiprows=range(14), header=None)
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.values
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

    graphset = RadiusGraph(10,data,label,task)

    return graphset



class XJTUGearboxRadius(object):
    num_classes = 9


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
            with open(os.path.join(self.data_dir, "XJTUGearboxRadius.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset
