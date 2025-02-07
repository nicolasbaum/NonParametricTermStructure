import numpy as np
import math
from sympy import Symbol, lambdify

def PhiFunction(t,j):
    return np.power(t,j)/np.math.factorial(j)

def getPhiBasisFunctions(p, start=0):
    def phi_j(x, j):
        x = np.asarray(x)            # force x to be array
        return np.power(x, j)/math.factorial(j)
    funcs = []
    for jidx in range(start, start+p):
        funcs.append(lambda arr, j=jidx: phi_j(arr, j))
    return funcs