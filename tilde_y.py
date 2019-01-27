import numpy as np
from N_operator import Nk
from D_operator import Dk

def tilde_yk(Pk, Fik, f0, tSpan, F=None):
    return Pk - Nk(Fik, f0, F, tSpan) + Dk(Fik, f0, f0, F, tSpan)

def tilde_y(P, Fi, f0, tSpan, F=None):
    result=np.empty(len(P))
    for k,Pk in enumerate(P):
        result[k]=tilde_yk(Pk, Fi[k], f0, tSpan, F)
    return result