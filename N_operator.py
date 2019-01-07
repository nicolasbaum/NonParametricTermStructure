import numpy as np
from matplotlib import pyplot as plt
from H_operator import Hi

def Nk(Fik, f, F, tRangeForBond):
    return np.cumsum(Fik * np.exp(-Hi(f, F, tRangeForBond)))