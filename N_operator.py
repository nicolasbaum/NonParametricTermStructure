import numpy as np
from matplotlib import pyplot as plt
from H_operator import Hi

def Nk( Fik, f, F=None, ti=1.0, dt=1.0):
    return np.sum( Fik*np.exp( -Hi(f,F,ti,dt) ) )