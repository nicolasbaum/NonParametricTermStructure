import numpy as np
from scalar_product import scalarProduct, getNormOfFunction

def getSigma(Fi, p, listOfksiFuncs, tSpan):
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds, numberOfBonds))
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            sij = scalarProduct(listOfksiFuncs[i], listOfksiFuncs[j], p, tSpan)
            result[i,j] = sij  # just store raw product
    return result