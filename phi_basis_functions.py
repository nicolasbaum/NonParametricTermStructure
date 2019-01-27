import numpy as np
from sympy import Symbol, lambdify

def PhiFunction(t,j):
    return np.power(t,j)/np.math.factorial(j)

def getPhiBasisFunctions(p,start=0):
    '''
    Given p, it should return a collection of basis functions of the form
    {(t^j)/j!} for j=0,...,p-1
    '''
    t = Symbol('t')
    return [ np.vectorize( lambdify(t, PhiFunction(t,j), "numpy") ) for j in range(start,p) ]