import numpy as np
from scalar_product import scalarProduct, getNormOfFunction

def getSigma(Fi, p, listOfksiFuncs, tSpan):
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds,numberOfBonds))
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            norm = getNormOfFunction(listOfksiFuncs[i],tSpan, p)*getNormOfFunction(listOfksiFuncs[j],tSpan, p)
            result[i,j]=scalarProduct(listOfksiFuncs[i], listOfksiFuncs[j], p, tSpan)/norm
    return result