import numpy as np
from scalar_product import scalarProduct
from calculate_ksi import ksi_k



def getSigma(Fi, f, F, p, tRange):
    listOfksiFuncs = ksiFuncs( Fi, f, F, p, tRange )
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds,numberOfBonds))
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            result[i,j]=scalarProduct(listOfksiFuncs[i],listOfksiFuncs[j],p,tRange)
    return result

def ksiFuncs( Fi, f, F, p, tRange):
    #p = Degree of derivative used in the smoothness equation
    ksiFuncs=[]
    #tRange = Symbol('tRange')
    for indexFi, Fik in enumerate(Fi):
        ksiFuncs.append( ksi_k(Fik, f, F, p, tRange ) )
    return ksiFuncs