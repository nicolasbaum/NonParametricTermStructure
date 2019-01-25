import numpy as np
from calculate_sigma import getSigma

def getM(Lambda,Fi,f,F,p,tSpan):
    Sigma = getSigma(Fi, f, F, p, tSpan)
    N=len(Fi)
    return Sigma+N*Lambda*np.eye(len(Sigma))