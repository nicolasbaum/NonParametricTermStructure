import numpy as np
from N_operator import Nk
from D_operator import Dk

def tilde_yk(Pk, Fik, f0, direction, tSpan, F=None):
    """
    Returns partial-lin:
       tilde_yk = Pk - Nk(Fik,f0,F,tSpan) + Dk(Fik,f0,directionFunc,F,tSpan).
    """
    return Pk - Nk(Fik, f0, F, tSpan) + Dk(Fik, f0, direction, F, tSpan)

def tilde_y(P, Fi, f0, direction, tSpan, F=None):
    """
    Build tilde_y for all bonds: shape (#bonds,).
    """
    result = np.empty(len(P))
    for k, Pk in enumerate(P):
        # Now we pass the same order:
        result[k] = tilde_yk(Pk, Fi[k], f0, direction, tSpan, F)
    return result