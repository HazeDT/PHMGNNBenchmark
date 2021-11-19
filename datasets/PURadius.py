import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.RadiusGraph import RadiusGraph
from datasets.AuxFunction import FFT
import pickle
#------------------------------------------------------------
signal_size=1024
root="D:\Data\Paderborn University_Bearing_Data"

#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']
label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 1-9
#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
label2=[6,7,8,9,10,11,12,13,14,15,16,17]    #The artificially damaged bearings data is labeled 4-15
#3 Bearings with real damages caused by accelerated lifetime tests(14x)
# RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
# label3=[18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #The artificially damaged bearings data is labeled 16-29
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']
label3=[i for i in range(13)]

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states

#generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []

    # for i in tqdm(range(len(HBdata))):
    #     name1 = state+"_"+HBdata[i]+"_1"
    #     path1=os.path.join('/tmp',root,HBdata[i],name1+".mat")        #_1----->1 can be replaced by the number between 1 and 20
    #     data1 = data_load(path1,name=name1,label=label1[i],InputType=InputType,task=task)
    #     data += data1

    #
    # for j in tqdm(range(len(ADBdata))):
    #     name2 = state+"_"+ADBdata[j]+"_1"
    #     path2=os.path.join('/tmp',root,ADBdata[j],name2+".mat")
    #     data2 = data_load(path2,name=name2,label=label2[j],InputType=InputType,task=task)
    #     data += data2


    for k in tqdm(range(len(RDBdata))):
        name3 = state+"_"+RDBdata[k]+"_1"
        path3 = os.path.join('/tmp',root,RDBdata[k],name3+".mat")
        data3 = data_load(sample_length, path3,name=name3,label=label3[k],InputType=InputType,task=task)
        data += data3


    return data

def data_load(signal_size, filename,name,label,InputType,task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data
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

    graphset = RadiusGraph(10,data,label,task)

    return graphset


#--------------------------------------------------------------------------------------------------------------------
class PURadius(object):
    num_classes = 13


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
            with open(os.path.join(self.data_dir, "PURadius.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset

