# encoding=utf-8
## Code of "Drift Compensation for an Electronic Nose by Adaptive Subspace Learning" Author: Dr.Liu
## Code recurrence
"""
    Modified on 2022 
    author: Xi Chen
"""
import numpy as np
from core import target_domain_scatter
from core import source_domain_scatter
from core import HSIC

def MyAlgorithm(Ms,Mt,batchS,batchT,Ys,alpha,lambdas,detla):
    '''
     calculate projection matrix P
    :param Ms: Source domain initial center
    :param Mt: Target domain initial center
    :param batchS: Source Data
    :param batchT: Target Data
    :param Ys: Source label
    :param alpha: Trade-off parameter
    :param lambdas: Trade-off parameter
    :param detla: Trade-off parameter
    :return: projection matrix P
    '''
    dim = np.size(batchS,1)                #  dimensional
    M = np.zeros((dim,dim))
    Ss = source_domain_scatter(batchS)     # source domain scatter matrix
    St = target_domain_scatter(batchT)     # target domain scatter matrix
    Ms = Ms.reshape((1,dim))
    Mt = Mt.reshape((1,dim))
    MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
    hsic = HSIC(batchS,Ys)                 # Hilbert-Schmidt independence criterion
    M = M + Ss + alpha*St - lambdas*MDD + detla*hsic # M matrix
    eigenvalue, eigenvector = np.linalg.eig(M)
    index = np.argsort(-eigenvalue)        # max index of eigenvalue
    neweigenvector = eigenvector[:,index]  # reoder
    return np.real(neweigenvector)