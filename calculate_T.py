import numpy as np
from D_operator import Dk
from phi_basis_functions import getPhiBasisFunctions


def _T_k(p, Fik, f0, F, tRangeForBond, T):
    '''
    T is a bidimensional matrix (k,p)
    This function returns T for kth bond
    '''
    result = np.zeros(p)
    phiBasis = getPhiBasisFunctions(p + 1, T)[1:]
    for phiIndex, phiFunc in enumerate(phiBasis):
        result[phiIndex] = Dk(Fik, f0, phiFunc, F, tRangeForBond)
    return result


def getT(p, Fi, f0, F, tSpan):
    result = np.zeros((len(Fi), p))
    T = tSpan[-1]
    for indexFi, Fik in enumerate(Fi):
        bondPeriods = int(np.nonzero(Fik)[0][-1]) + 1
        tRangeForBond = tSpan[:bondPeriods]
        result[indexFi, :] = _T_k(p, Fik[:bondPeriods], f0, F, tRangeForBond, T)
    return result
