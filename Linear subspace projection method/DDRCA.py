# encoding=utf-8
"""
    Modified on 2022 
    author: Xi Chen
"""

import numpy as np
from core import within_class_scatter
from core import target_domain_scatter
from core import source_domain_scatter
from core import between_class_scatter

def DDRCA(Ms,Mt,batchS,batchT,Ys,alpha,lambdas,detla):

    dim = np.size(batchS, 1)
    Ns = np.size(batchS,0)
    Nt = np.size(batchT,0)
    M = np.zeros((dim, dim))
    Ss = source_domain_scatter(batchS)
    Sw = within_class_scatter(batchS, Ys)
    St = target_domain_scatter(batchT)
    Sb = between_class_scatter(batchS,Ys)
    Ms = Ms.reshape((1, dim))  # reshape
    Mt = Mt.reshape((1, dim))  # reshape
    MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
    tem = np.linalg.pinv(MDD)
    tem1 = Ss/Ns + (alpha*St)/Nt - lambdas*Sw + detla*Sb
    M = M + np.dot(tem,tem1)
    eigenvalue, eigenvector = np.linalg.eig(M)
    eigenvalue = np.real(eigenvalue)
    Idex = np.argsort(-eigenvalue)
    eigenvector = np.real(eigenvector)
    neweigenvector = eigenvector[:, Idex]
    return neweigenvector


