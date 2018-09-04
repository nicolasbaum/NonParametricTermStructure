import numpy as np
from matplotlib import pyplot as plt

def Hi(f, tRange, F=None):
    F = F or (lambda x:x*x)
    F = np.vectorize( F )
    f = np.vectorize( f )
    return np.trapz( F( f(tRange) ), tRange )