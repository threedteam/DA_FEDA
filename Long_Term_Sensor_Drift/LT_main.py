import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from Sensor_GAN.BP_model import FeatureExtractor,class_classifier,Domain_classifier
from Sensor_GAN.BP_loaddataset import train_loader,test_loader
from Sensor_GAN import BP_train,BP_test,BP_params
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rc('font',family='Times New Roman')
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from time import sleep
import argparse,sys,os



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

def ToOne(dataSet):                                     
    xmin=dataSet.min(axis=0)
    xmax=dataSet.max(axis=0)
    std=(dataSet-xmin)/(xmax-xmin)
    return std

def plot(feature_extractor,source_data,source_label,target_data,target_label):
    """
    绘制数据降维的图像
    :param feature_extractor:
    :param source_data:
    :param target_data:
    :return:
    """
    feature_extractor.eval()
    input1 = torch.FloatTensor(source_data)
    feature1 = feature_extractor(input1)
    feature1 = feature1.detach().numpy()

    input2 = torch.FloatTensor(target_data)
    feature2 = feature_extractor(input2)
    feature2 = feature2.detach().numpy()

    plt.figure(figsize=(12, 18))
    plt.subplots_adjust(wspace=0.35)
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 2, 1)
    data, label = feature1, source_label
    scaler = PCA(n_components=2)
    data_1 = scaler.fit_transform(data)
    data_1 = pd.DataFrame(data_1)
    data = data_1.values

    index_1 = np.where(label == 1)
    plt.scatter(data[index_1, 0], data[index_1, 1], marker='x', color='r', label='Class 1', s=20)
    index_2 = np.where(label == 2)
    plt.scatter(data[index_2, 0], data[index_2, 1], marker='o', color='r', label='Class 2', s=20)
    index_3 = np.where(label == 3)
    plt.scatter(data[index_3, 0], data[index_3, 1], marker='+', color='r', label='Class 3', s=20)
    index_4 = np.where(label == 4)
    plt.scatter(data[index_4, 0], data[index_4, 1], marker='*', color='r', label='Class 4', s=20)
    index_5 = np.where(label == 5)
    plt.scatter(data[index_5, 0], data[index_5, 1], marker='s', color='r', label='Class 5', s=20)
    index_6 = np.where(label == 6)
    plt.scatter(data[index_6, 0], data[index_6, 1], marker='1', color='r', label='Class 6', s=20)
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.title('Source:Batch 1', fontsize=20)

    plt.subplot(2, 2, 2)
    scaler1 = PCA(n_components=2)
    data1, label1 = feature2, target_label
    data_2 = scaler1.fit_transform(data1)
    data_2 = pd.DataFrame(data_2)
    data = data_2.values
    index_1 = np.where(label == 1)
    plt.scatter(data[index_1, 0], data[index_1, 1], marker='x', color='b', label='Class 1', s=20)
    index_2 = np.where(label == 2)
    plt.scatter(data[index_2, 0], data[index_2, 1], marker='o', color='b', label='Class 2', s=20)
    index_3 = np.where(label == 3)
    plt.scatter(data[index_3, 0], data[index_3, 1], marker='+', color='b', label='Class 3', s=20)
    index_4 = np.where(label == 4)
    plt.scatter(data[index_4, 0], data[index_4, 1], marker='*', color='b', label='Class 4', s=20)
    index_5 = np.where(label == 5)
    plt.scatter(data[index_5, 0], data[index_5, 1], marker='s', color='b', label='Class 5', s=20)
    index_6 = np.where(label == 6)
    plt.scatter(data[index_6, 0], data[index_6, 1], marker='1', color='b', label='Class 6', s=20)
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.title('Target:Batch 2', fontsize=20)
    save_path = 'F:/Deep_Learning/Sensor_GAN/大论文实验/'
    fig = plt.gcf()
    fig.set_size_inches((20, 11), forward=False)
    fig.savefig(save_path + "ceshi" + '.png', dpi=300)

def main(train_mode,FeatureExtractor,class_classifier,Domain_classifier,train_loader,test_loader):
    # BP_params.fig_mode = args.fig_mode
    # BP_params.epochs = args.max_epoch
    # BP_params.train_mode = args.train_mode
    # BP_params.source_domain = args.source_domain
    # BP_params.target_domain = args.target_domain

    src_train_dataloader = train_loader
    src_test_dataloader = train_loader
    tgt_test_dataloader = test_loader
    tgt_trian_dataloader = test_loader

    feature_extractor = FeatureExtractor
    class_classifier = class_classifier
    domain_classifier = Domain_classifier

    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()
    # class_criterion = MultiCEFocalLoss(6)


    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                           {'params': class_classifier.parameters()},
                           {'params': domain_classifier.parameters()}], lr=BP_params.lr, momentum=0.9)
    for epoch in range(BP_params.epochs):
        print("Epoch: {}".format(epoch))
        BP_train.train(train_mode,feature_extractor,class_classifier,domain_classifier,class_criterion,domain_criterion,
                       src_train_dataloader,tgt_trian_dataloader,optimizer,epoch)
        accuracy = BP_test.test(feature_extractor,class_classifier,domain_classifier,src_test_dataloader,tgt_test_dataloader)
        
        print("##################################")
        print(accuracy)
        print("###################################")

if __name__ == '__main__':
    train_mode = "dann"
    feature_extractor,class_classifier,domain_classifier = FeatureExtractor(),class_classifier(),Domain_classifier()
    train_loader,test_loader=train_loader,test_loader
    main(train_mode,feature_extractor,class_classifier,domain_classifier,train_loader,test_loader)

