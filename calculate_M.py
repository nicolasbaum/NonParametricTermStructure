import numpy as np
from calculate_sigma import sigma

def M(Lambda,Fi,f,F,p,tSpan):
    Sigma = sigma(Fi, f, F, p, tSpan)
    N=len(Fi)
    return Sigma+N*Lambda*np.eye(Sigma.shape)