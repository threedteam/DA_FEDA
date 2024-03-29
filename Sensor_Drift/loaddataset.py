import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable

"""
This is a demo
"""

# def get_train_loader(dataset):
#     """
#     Get train dataloader of source domain or target domain
#     :return: dataloader
#     """
#     if dataset == ' ':

#         data = datasets.
#         dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
#     elif dataset == ' ':
#         data = datasets.
#         dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
#     else:
#         raise Exception('There is no dataset named {}'.format(str(dataset)))
#     return dataloader



# def get_test_loader(dataset):
#     """
#     Get test dataloader of source domain or target domain
#     :return: dataloader
#     """
#     if dataset == ' ':
#         data = datasets.
#         dataloader = DataLoader(dataset= data, batch_size= 1, shuffle= False)
#     elif dataset == ' ':
#         data = datasets.
#         dataloader = DataLoader(dataset = data, batch_size= 1, shuffle= False)
#     else:
#         raise Exception('There is no dataset named {}'.format(str(dataset)))
#     return dataloader


def datapreprocessing(data):     ## data preprocessing , input data format xxx.csv
    '''
    data preprocessing
    :param data:
    :return:
    '''
    dataset = data.iloc[:,:-1]        ## obtain the dataset without label, the data format is data + label (the labels are in the last column) 
    label = data.iloc[:,-1].values    ## obtain the label
    data_temp = dataset.values        ## .csv to numpy 
    dataset = (dataset/np.sqrt(sum(data_temp*data_temp))).values  ## L2 norm regularization
    return dataset,label              ## return dataset and label

## Different experimental batches need to load different experimental data
dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch1.csv',header=None)  ## load source data 
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch2.csv",header=None)  ## load target data
batchS,batchS_label = datapreprocessing(dataS)                                                     ## data preprocessing
batchT,batchT_label = datapreprocessing(dataT)                                                     ## data preprocessing
# batchS,batchS_label = datapreprocessing(dataS)
# batchT,batchT_label = datapreprocessing(dataT)

input_data = torch.FloatTensor(batchS)                                                             ## The format is converted to tensor
label = torch.LongTensor(batchS_label-1)                                                           ## The format is converted to tensor

input_data1 = torch.FloatTensor(batchT)
label1 = torch.LongTensor(batchT_label-1)

train_dataset = Data.TensorDataset(input_data,label)                                              ## Convert to PyTorch's unique data format
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)                  

test_dataset = Data.TensorDataset(input_data1,label1)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)
