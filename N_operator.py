import numpy as np
from matplotlib import pyplot as plt
from H_operator import Hi

def Nk(Fik, f, F, tRange):
    return np.sum(Fik * np.exp(-Hi(f, F, tRange)))