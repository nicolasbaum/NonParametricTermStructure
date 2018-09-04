import numpy as np
from sympy import Symbol, lambdify
from numba import jit

def PhiFunction(t,j):
    return np.power(t,j)/np.math.factorial(j)

@jit
def getPhiBasisFunctions(p,start=0):
    '''
    Given p, it should return a collection of basis functions of the form
    {(t^j)/j!} for j=0,...,p-1
    '''
    t = Symbol('t')
    phiBasisFunctions = []
    p=p+start if start>0 else p
    for j in range(start,p):
        phiBasisFunctions.append( lambdify(t, PhiFunction(t,j), t, "numpy") )

    return phiBasisFunctions