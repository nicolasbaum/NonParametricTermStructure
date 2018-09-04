import numpy as np
from matplotlib import pyplot as plt
from H_operator import Hi

def Nk(Fik, f, tRange, F=None):
    return np.sum(Fik * np.exp(-Hi(f, tRange, F)))