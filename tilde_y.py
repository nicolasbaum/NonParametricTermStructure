import numpy as np
from N_operator import Nk
from D_operator import Dk

def tilde_y(Pk, Fik, f0, f, tRange, F=None):
    return Pk - Nk(Fik, f, tRange, F) + Dk(Fik, f0, f, tRange, F)