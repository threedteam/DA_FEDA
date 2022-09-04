You need to install the PyTorch, numpy, pandas, etc. libraries of Python

########################

######
python                     3.7.7
numpy                     1.18.5
pandas                     1.0.5

#######################

Pytorch version,  GPU or not？
>>>>
import torch
print(torch.__version__)  
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

>>>>
1.5.1
True
GeForce GTX 1660
#######################
 

Read along with the source code
There are relevant annotation in the source code

############################################################
############################################################

######
The following py files are auxiliary files
##############################

########
LT_model.py
This py file is mainly code for feature extractor, discriminator, gradient reversal layer, label classifier


1. class GradReverse
 
gradient reversal layer, it's used to reverse the gradient.

2. class FeatureExtractor

A standard feedforward neural network is constructed as the feature extractor to extract  domain-invariant feature，set the size of the input layer
unit according to the specific data format

3. class_classifier

class classifier is used to classify labels, set the size of the output layer unit according to the specific data format

4. class Domain_classifier

discriminator is used to discriminate feature that come from which domain

##########################################

######
LT_loaddataset.py

This py file is main code for loading data
Different experimental batches need to load different experimental data
please see the code annotation in the source code

##########################################

#######
LT_params.py

Parameter setting, please refer to the paper for the specific parameter meaning

###########################################

######
LT_utils.py

This py file is main code to adjust the learning rate of optimizer
please see the code annotation in the source code

#####################################################################
#####################################################################

######
The following py files are core files
###########################

######
LT_train.py
 
This py file is main code for training.
please see the code annotation in the source code

1. First, load some specific Python libraries

2. The function of  "get_L2Norm_loss (x)" is used to calculate the feature norm loss L_f, redius is the $Z_{x_i}^*$, dynamic way

3. The function of "get_L2norm_loss_self_driven(x)" is used to calculate the feature loss L_z,  redius is a fixed value ( see the LT_params.py file ), fixed way

4. The function of "get_entropy_loss(x)" is used to calculate the  conditional entropy of target domain 

5. The function of "train (training_mode,feature_extractor,class_classifier,domain_classifier,class_criterion,domain_criterion,
                                       source_dataloader,target_dataloader,optimizer,epoch)" is the main function.
   Please see the code annotation in the source code

########################################################################

#######
LT_test.py

This py file is main code for testing
please see the code annotation in the source code

######################################################################

######
LT_main.py

The main purpose of this py file is to concatenate all the py files
please see the code annotation in the source code