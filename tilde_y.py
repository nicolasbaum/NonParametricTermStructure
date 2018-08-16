import numpy as np
from N_operator import Nk
from D_operator import Dk

def tilde_y(Pk, Fi, f0, f, F=None, ti=1, dt=0.001):
    return Pk - Nk( Fi, f, F, ti, dt)+Dk( Fi, f0, f, F, ti, dt )