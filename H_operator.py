import numpy as np
from matplotlib import pyplot as plt

def Hi(f, F, tRange):
    return np.trapz( F( f(tRange) ), tRange )