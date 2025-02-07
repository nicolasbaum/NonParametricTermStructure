import numpy as np
from D_operator import Dk
from phi_basis_functions import getPhiBasisFunctions

def _T_k(p, Fik, f0, F, tRangeForBond):
    """
    Build row T_k = [Dk(Fik, f0, phi_0), ..., Dk(Fik, f0, phi_{p-1})].
    """
    T = np.zeros(p)
    # was: phiBasis = getPhiBasisFunctions(p+1, start=1)
    # now:
    phiBasis = getPhiBasisFunctions(p, start=0)

    for j, phiFunc in enumerate(phiBasis):
        T[j] = Dk(Fik, f0, phiFunc, F, tRangeForBond)
    return T

def getT(p, Fi, f0, F, tSpan):
    result = np.zeros((len(Fi), p))
    for indexFi, Fik in enumerate(Fi):
        bondPeriods = int(np.nonzero(Fik)[0][-1]) + 1
        tRangeForBond = tSpan[:bondPeriods]
        result[indexFi,:] = _T_k(p, Fik[:bondPeriods], f0, F, tRangeForBond)
    return result
