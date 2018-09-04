import numpy as np
from scalar_product import scalarProduct
from sympy import Symbol, lambdify
from calculate_ksi import ksi_k



def sigma( Fi, f, F, p, tSpan ):
    listOfksiFuncs = ksiFuncs( Fi, f, F, p )
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds,numberOfBonds))
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            result[i,j]=scalarProduct(listOfksiFuncs[i],listOfksiFuncs[i],p,tSpan)
    return result

def ksiFuncs( Fi, f, F, p ):
    ksiFuncs=[]
    t = Symbol('t')
    for indexFi, Fik in enumerate(Fi):
        ksiFuncs.append( lambdify(t, ksi_k(Fik, f, F, p, t), t, "numpy") )
    return ksiFuncs