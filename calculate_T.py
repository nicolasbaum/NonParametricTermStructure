import numpy as np
from D_operator import Dk
from phi_basis_functions import getPhiBasisFunctions

def _T_k(p, Fik, f0, F, tRangeForBond):
    '''
    T is a bidimensional matrix (k,p)
    This function returns T for kth bond
    '''
    T = np.zeros(p)
    phiBasis = getPhiBasisFunctions(p+1,start=1)
    for phiIndex, phiFunc in enumerate(phiBasis):
        T[phiIndex] = Dk(Fik, f0, phiFunc, F, tRangeForBond)
    return T

def getT(p, Fi, f0, F, tSpan):
    result=np.zeros((len(Fi),p))
    for indexFi, Fik in enumerate(Fi):
        bondPeriods = int(np.nonzero(Fik)[0][-1]) + 1
        tRangeForBond = tSpan[1:bondPeriods+1]
        result[indexFi,:]=_T_k(p, Fik[:bondPeriods], f0, F, tRangeForBond)
    return result
