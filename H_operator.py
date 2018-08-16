import numpy as np
from matplotlib import pyplot as plt

def Hi( f, F=None, ti=1, dt=0.001):
    tRange = np.linspace(0,ti,int(ti/dt))
    F = F or (lambda x:x*x)
    F = np.vectorize( F )
    f = np.vectorize( f )
    return np.trapz( F( f(tRange) ), tRange )