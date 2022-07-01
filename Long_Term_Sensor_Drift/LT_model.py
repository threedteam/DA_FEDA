import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer  
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x


    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Linear(128,200)
        self.batch = nn.BatchNorm1d(200)


    def forward(self,x):
        x = F.relu(self.feature(x))
        x = self.batch(x)
        return x

class class_classifier(nn.Module):
    def __init__(self):
        super(class_classifier, self).__init__()
        self.out = nn.Linear(200,6)

    def forward(self,x):
        out1 = self.out(x)
        return F.log_softmax(out1,1),out1

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.domian_out1 = nn.Linear(200,100)
        self.domian_out2 = nn.Linear(200,2)

    def forward(self,x,constant):
        input = GradReverse.grad_reverse(x,constant)
        # input = F.relu(self.domian_out1(input))
        return F.log_softmax(self.domian_out2(input),1)









# examples = enumerate(zip(train_loader,test_loader))
# batch_idx,(sdata,tdata) = next(examples)
#
# input1,label1 = sdata
# input2,label2 = tdata
#
# print(input1.shape[0],input2.shape[0])
# fea = FeatureExtractor()
# c_lass = class_classifier()
# d_c = Domain_classifier()
# out1 = fea(input1)
# out2 = fea(input2)
#
# tar_sou = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))
# cre = nn.CrossEntropyLoss()
# preds = c_lass(out1)
# loss = cre(preds,label1)
# pre1 = d_c(out2,2)
# preds = preds.data.max(1,keepdim=True)[1]
# loss1 = cre(pre1,tar_sou)
# print(preds.eq(label1.data.view_as(preds)).cpu().sum())





#
# input_data = torch.FloatTensor(batchS)
# extractor = FeatureExtractor()
# classifier = class_classifier()
# domain = Domain_classifier()
#
# fea = extractor(input_data)
# out = classifier(fea)
# out1 = domain(fea,1)

