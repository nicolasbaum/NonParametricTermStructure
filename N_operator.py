import numpy as np
from H_operator import Hi

def Nk(Fik, f, F, tRangeForBond):
    return np.sum(Fik[:len(tRangeForBond)] * np.exp(-Hi(f, F, tRangeForBond)))