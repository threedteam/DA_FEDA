"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np


def test(feature_extractor,class_classifier,domain_classifier,source_dataloader,target_dataloader):
    """
    测试模型
    :param feature_extractor:  从目标域中提取特征
    :param class_classifier:   预测类别标签
    :param domain_classifier:  预测域类别标签
    :param source_dataloader:  源域数据加载
    :param target_loader:      目标域数据加载
    :return:
    """
    
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    tgt_correct = 0.0
    src_correct = 0.0

   
    for batch_idx,sdata in enumerate(source_dataloader):
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10*p)) - 1

        input1,label1 = sdata
        src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))

        output1,_= class_classifier(feature_extractor(input1))
        pred1 = output1.data.max(1,keepdim=True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

        src_preds = domain_classifier(feature_extractor(input1),constant)
        src_preds = src_preds.data.max(1,keepdim=True)[1]
        src_correct += src_preds.eq(src_labels.data.view_as(src_preds)).cpu().sum()

    
    for batch_idx,tdata in enumerate(target_dataloader):
        p = float(batch_idx) / len(target_dataloader)
        constant = 2. / (1. + np.exp(-10*p)) - 1

        input2,label2 = tdata
        tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        output2,_ = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1,keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

        tgt_preds = domain_classifier(feature_extractor(input2),constant)
        tgt_preds = tgt_preds.data.max(1,keepdim=True)[1]
        tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()

    domain_correct = tgt_correct + src_correct
    target_accuracy = 100. * float(target_correct)/len(target_dataloader.dataset)
    print("\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} , acc{:.4f}%\n"
          "Domain Accuracy: {}/{} ({:.4f}%)\n".format
          (source_correct,len(source_dataloader.dataset),100. * float(source_correct)/len(source_dataloader.dataset),
           target_correct,len(target_dataloader.dataset),100. * float(target_correct)/len(target_dataloader.dataset),
           domain_correct,len(source_dataloader.dataset)+len(target_dataloader.dataset),
           100. * float(domain_correct) / (len(target_dataloader.dataset)+len(source_dataloader.dataset))))
    return target_accuracy
