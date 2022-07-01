import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import params
from sklearn.manifold import TSNE

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer 调整优化器的学习率
    :param optimizer: optimizer for updating parameters  用于参数更新的优化器
    :param p: a variable for adjusting learning rate     用于调整学习速率的变量
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer
