import numpy as np
from D_operator import Dk
from phi_basis_functions import getPhiBasisFunctions

def T_k(p,Fik,f0,F,tRange):
    '''
    T is a bidimensional matrix (k,p)
    This function returns T for kth bond
    '''
    T = np.zeros(p)
    phiBasis = getPhiBasisFunctions(p,start=1)
    for phiIndex, phiFunc in enumerate(phiBasis):
        T[phiIndex] = Dk(Fik, f0, phiFunc, tRange, F)
    return T