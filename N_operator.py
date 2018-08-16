import numpy as np
from matplotlib import pyplot as plt
from H_operator import Hi

def Nk( Fi, f, F=None, ti=1, dt=0.001):
    return np.sum( Fi*np.exp( -Hi(f,F,ti,dt) ) )