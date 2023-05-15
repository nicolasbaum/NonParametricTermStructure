import numpy as np
from scalar_product import scalarProduct


def getSigma(Fi, p, listOfksiFuncs, tSpan):
    numberOfBonds = len(Fi)
    result = np.zeros((numberOfBonds, numberOfBonds))
    calculations = numberOfBonds ** 2
    for i in range(numberOfBonds):
        for j in range(numberOfBonds):
            result[i, j] = scalarProduct(listOfksiFuncs[i], listOfksiFuncs[j], p, tSpan[-1])
            calculations -= 1
            print(f"Sigma[{i},{j}]. Pending {calculations} calculations")
    return result
