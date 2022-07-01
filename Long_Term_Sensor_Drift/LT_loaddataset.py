import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable

def datapreprocessing(data):
    '''
    data preprocessing
    :param data:
    :return:
    '''
    dataset = data.iloc[:,:-1]
    label = data.iloc[:,-1].values
    data_temp = dataset.values
    dataset = (dataset/np.sqrt(sum(data_temp*data_temp))).values
    return dataset,label

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch1.csv',header=None)
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch2.csv",header=None)
batchS,batchS_label = datapreprocessing(dataS)
batchT,batchT_label = datapreprocessing(dataT)
# batchS,batchS_label = datapreprocessing(dataS)
# batchT,batchT_label = datapreprocessing(dataT)

input_data = torch.FloatTensor(batchS)
label = torch.LongTensor(batchS_label-1)

input_data1 = torch.FloatTensor(batchT)
label1 = torch.LongTensor(batchT_label-1)

train_dataset = Data.TensorDataset(input_data,label)
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

test_dataset = Data.TensorDataset(input_data1,label1)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)