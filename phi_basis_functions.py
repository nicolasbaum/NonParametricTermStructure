import numpy as np
from scalar_product import scalarProduct
from sympy import Symbol, lambdify

def PhiFunction(t,j):
    return np.power(t,j)/np.math.factorial(j)

def getPhiBasisFunctions(p,tRange,start=0):
    '''
    Given p, it should return a collection of basis functions of the form
    {(t^j)/j!} for j=0,...,p-1
    '''
    t = Symbol('t')
    phiBasisFunctions = []
    p=p+start if start>0 else p
    for j in range(start,p):
        phiFunc = lambdify(t, PhiFunction(t,j), "numpy")
        norm = scalarProduct(phiFunc, phiFunc, p,tRange)
        phiBasisFunctions.append( lambda t: phiFunc(t)/norm )

    return phiBasisFunctions