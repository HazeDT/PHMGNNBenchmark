import os
import numpy as np
from itertools import islice
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.RadiusGraph import RadiusGraph
from datasets.AuxFunction import FFT
import pickle
#------------------------------------------------------------
signal_size=1024

#Data names of 5 bearing fault types under two working conditions
Bdata = ["ball_20_0.csv","comb_20_0.csv","health_20_0.csv","inner_20_0.csv","outer_20_0.csv","ball_30_2.csv","comb_30_2.csv","health_30_2.csv","inner_30_2.csv","outer_30_2.csv"]
label1 = [i for i in range(0,10)]
#Data names of 5 gear fault types under two working conditions
Gdata = ["Chipped_20_0.csv","Health_20_0.csv","Miss_20_0.csv","Root_20_0.csv","Surface_20_0.csv","Chipped_30_2.csv","Health_30_2.csv","Miss_30_2.csv","Root_30_2.csv","Surface_30_2.csv"]
labe12 = [i for i in range(10,20)]


#generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    datasetname:List of  dataset
    '''
    datasetname = os.listdir(os.path.join(root, os.listdir(root)[2]))  # 0:bearingset, 2:gearset
    root1 = os.path.join("/tmp",root,os.listdir(root)[2],datasetname[0]) #Path of bearingset
    root2 = os.path.join("/tmp",root,os.listdir(root)[2],datasetname[2]) #Path of gearset

    data = []

    for i in tqdm(range(len(Bdata))):
        path1 = os.path.join('/tmp',root1,Bdata[i])
        data1 = data_load(sample_length, path1,dataname=Bdata[i],label=label1[i],InputType=InputType,task=task)
        data += data1


    for j in tqdm(range(len(Gdata))):
        path2 = os.path.join('/tmp',root2,Gdata[j])
        data2 = data_load(sample_length, path2,dataname=Gdata[j],label=labe12[j],InputType=InputType,task=task)
        data += data2


    return data

def data_load(signal_size,filename,dataname,label,InputType,task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    f = open(filename,"r",encoding='gb18030',errors='ignore')
    fl=[]
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",",8)   #Separated by commas
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t",8)   #Separated by \t
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    fl = np.array(fl)
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1,)
    data=[]
    start,end=0,signal_size
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

    graphset = RadiusGraph(10, data, label, task)

    return graphset



#--------------------------------------------------------------------------------------------------------------------
class SEURadius(object):
    num_classes = 20


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
            with open(os.path.join(self.data_dir, "SEURadius.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset

