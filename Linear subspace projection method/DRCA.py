# encoding=utf-8
## Code of "Anti-drift in e-nose: A subspace projection approach with drift reduction" Author: Dr.Zhang
## Code recurrence
"""
    Modified on 2022 
    author: Xi Chen
"""
import numpy as np
from core import source_domain_scatter
from core import target_domain_scatter

def DRCA(Ms,Mt,batchS,batchT,Lambda):
    """
    calculate projection matrix P
    :param Ms: Source domain initial center
    :param Mt: Target domain initial center
    :param batchS: Source Data
    :param batchT: Target Data
    :param Lambda:Trade-off parameter
    :return: projection matrix P
    """
    dim = np.size(batchS,1)
    A = np.zeros((dim,dim))
    Ms = Ms.reshape((1, dim))  # reshape
    Mt = Mt.reshape((1, dim))  # reshape
    tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
    Ss = source_domain_scatter(batchS)  # source domain scatter matrix
    St = target_domain_scatter(batchT)  # target domain scatter matrix
    tem1 = Ss + Lambda*St
    A = A + np.dot(tem,tem1)
    eigenvalue,eigenvector = np.linalg.eig(A)
    eigenvalue = np.real(eigenvalue)
    Idex = np.argsort(-eigenvalue)
    eigenvector = np.real(eigenvector)
    neweigenvector = eigenvector[:,Idex]
    return neweigenvector













