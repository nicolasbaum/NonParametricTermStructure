import numpy as np
from scalar_product import scalarProduct


def getSigma(Fi, p, listOfksiFuncs, tSpan):
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds, numberOfBonds))
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            result[i, j] = scalarProduct(listOfksiFuncs[i], listOfksiFuncs[j], p, tSpan[-1])
    return result
