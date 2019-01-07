import numpy as np
from N_operator import Nk
from D_operator import Dk

def tilde_yk(Pk, Fik, f0, f, tRange, F=None):
    return Pk - Nk(Fik, f, F, tRange) + Dk(Fik, f0, f, F, tRange)