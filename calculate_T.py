import numpy as np
from D_operator import Dk
from phi_basis_functions import getPhiBasisFunctions

def _T_k(p, Fik, f0, F, tRange):
    '''
    T is a bidimensional matrix (k,p)
    This function returns T for kth bond
    '''
    T = np.zeros(p)
    phiBasis = getPhiBasisFunctions(p,start=1)
    for phiIndex, phiFunc in enumerate(phiBasis):
        T[phiIndex] = Dk(Fik, f0, phiFunc, F, tRange)
    return T

def T(p,Fi,f0,F,tSpan):
    result=np.zeros((len(Fi),p))
    for indexFi, Fik in enumerate(Fi):
        tRange=tSpan[:len(Fik)]
        result[indexFi,:]=_T_k(p, Fik, f0, F, tRange)
    return result
