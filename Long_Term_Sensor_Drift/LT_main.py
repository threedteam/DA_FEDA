import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from Long_Term_Sensor_Drift.LT_model import FeatureExtractor,class_classifier,Domain_classifier
from Long_Term_Sensor_Drift.LT_loaddataset import train_loader,test_loader
from Long_Term_Sensor_Drift import LT_train,LT_test,LT_params
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rc('font',family='Times New Roman')
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from time import sleep
import argparse,sys,os


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

