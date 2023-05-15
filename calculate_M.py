import numpy as np
from calculate_sigma import getSigma


def getM(Lambda, Fi, listOfksiFuncs, p, tSpan):
    Sigma = getSigma(Fi, p, listOfksiFuncs, tSpan)
    N = len(Fi)
    return Sigma + N * Lambda * np.eye(len(Sigma))
