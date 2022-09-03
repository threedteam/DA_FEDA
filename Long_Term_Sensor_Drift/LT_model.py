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
        grad_output = grad_output.neg() * ctx.constant     ## .neg(), Returns the input tensor as negative by element, out=−1∗input
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Linear(128,200)               ## input layer, according to the data format
        self.batch = nn.BatchNorm1d(200)                ## normalization


    def forward(self,x):
        x = F.relu(self.feature(x))                     ## The activation function relu
        x = self.batch(x)
        return x

class class_classifier(nn.Module):
    def __init__(self):
        super(class_classifier, self).__init__()
        self.out = nn.Linear(200,6)      ## Sets the output layer units according to the data format

    def forward(self,x):
        out1 = self.out(x)
        return F.log_softmax(out1,1),out1   ## output the label and probability

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.domian_out1 = nn.Linear(200,100)
        self.domian_out2 = nn.Linear(200,2)      ## domain label source domain as 0 and target as 1

    def forward(self,x,constant):
        input = GradReverse.grad_reverse(x,constant)  ##  reverse the gradient.
        # input = F.relu(self.domian_out1(input))
        return F.log_softmax(self.domian_out2(input),1)



