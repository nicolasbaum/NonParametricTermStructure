import numpy as np
from H_operator import Hi

def dVectorFromf( f, F, tSpan ):
    return np.exp(-Hi(f,F,tSpan))

def zVectorFromf( f, F, tSpan):
    return Hi(f,F,tSpan)