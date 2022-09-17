import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from Sensor_Drift import params
from Sensor_Drift import utils
import warnings
warnings.filterwarnings("ignore")

"""
This is a demo
"""

# Calculate the L_f
def get_L2Norm_loss(x):
    redius = x.norm(p=2,dim=1).detach()
    assert redius.requires_grad == False
    redius +=redius + 1.0
    l = ((x.norm(p=2,dim=1)-redius)**2).mean()
    return params.weight_L2norm*l

# Calculate the L_z
def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2,dim=1).mean()-params.redius)**2
    return params.weight_ring*l

# Calculate the conditional entropy of target domain
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax,mask)
    entropy = -(torch.sum(mask_out*torch.log(mask_out)))
    return params.weight_entropy*(entropy/float(p_softmax.size(0)))

def train(training_mode,feature_extractor,class_classifier,domain_classifier,class_criterion,domain_criterion,
          source_dataloader,target_dataloader,optimizer,epoch):
    """
    training mode
    :param training_mode:  
    :param feature_extractor: feature extractor G_f
    :param class_classifier: label classifier G_y
    :param domain_classifier: discriminator G_d
    :param class_criterion: label classifier loss function
    :param domain_criterion: domain adversarial loss function
    :param source_dataloader: source data
    :param target_dataloader: target data
    :param optimizer: 
    :param epoch: the number of iterations
    :return:
    """
    # Set the model to training mode
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # Set the experimental step
    start_steps = epoch*len(source_dataloader)          ## start step
    total_steps = params.epochs * len(source_dataloader)  ## total steps

    for batch_idx,(sdata,tdata) in enumerate(zip(source_dataloader,target_dataloader)):  ## loop training, the standard PyTorch training mode
        if training_mode == "dann" : # training model is set to dann
            # Setting HyperParameters
            p = float(batch_idx + start_steps) / total_steps     ##  a variable for adjusting learning rate 
            constant = 2./(1+np.exp(-params.gamma*p))-1       ##  a constant of RevGrad

            # Get data for the source and target domains
            input1,label1 = sdata
            input2,label2 = tdata

            # Dynamically adjust the learning rate, the standard format for PyTorch
            optimizer = utils.optimizer_scheduler(optimizer,p)
            optimizer.zero_grad()                                   

            # Set the domain label to 0 for the source domain and 1 for the target domain
            source_labels = Variable(torch.zeros(input1.size()[0])).type(torch.LongTensor)   ##  source label 0   (domain label)
            target_labels = Variable(torch.ones(input2.size()[0])).type(torch.LongTensor)    ##  target label 1   (domain label)

            ## Feature extraction using feature extractor 
            src_feature = feature_extractor(input1)    ## source
            tgt_feature = feature_extractor(input2)    ## target

            # compute the class loss of src_feature  
            class_preds,s_logits= class_classifier(src_feature)   # Predicted labels and probabilities
            class_loss = class_criterion(class_preds, label1)     ## Calculate the cross-loss of the source domain

            # Calculate the L2 norm
            src_l2_loss = get_L2Norm_loss(src_feature)  # source
            tgt_l2_loss = get_L2Norm_loss(tgt_feature)  # target
            
            # Calculate the entropy of the target domain
            _,t_logits = class_classifier(tgt_feature)
            t_probs = F.softmax(t_logits)
            t_entropy_loss = get_entropy_loss(t_probs)
            %s_probs = F.softmax(s_logits)
            %s_entropy_loss = get_entropy_loss(s_probs)
            %entropy_loss = s_entropy_loss + t_entropy_loss
            
            # Compute domain adversarial losses
            tgt_preds = domain_classifier(tgt_feature,constant)
            src_preds = domain_classifier(src_feature,constant)
            tgt_loss = domain_criterion(tgt_preds,target_labels)   ## target loss
            src_loss = domain_criterion(src_preds,source_labels)   ## source loss
            domain_loss = tgt_loss + src_loss                      ## domain loss = target adversarial loss + source adversarial loss  

            loss = class_loss + params.theta*domain_loss + src_l2_loss + tgt_l2_loss + t_entropy_loss  ## total loss
            loss.backward()                # the standard PyTorch training mode     
            optimizer.step()               # the standard PyTorch training mode

            # Print training process
            # According to the need to choose
            
            
            # if (batch_idx + 1) % 2 == 0:
            #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
            #         batch_idx * len(input2), len(target_dataloader.dataset),
            #         100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
            #         domain_loss.item()
            #     ))












