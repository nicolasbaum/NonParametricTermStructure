import numpy as np
from matplotlib import pyplot as plt
from optimization import optCumTrapz
from N_operator import Nk
from sympy import Symbol, diff, lambdify

def Dk(Fik, f0, f, F, tRange):
    x = Symbol('x')
    #Revisar esto de la diferenciacion de F
    diffedF = lambdify( x, diff( F(x), x ), "numpy" )
    return -1 * Nk(Fik, f0, F, tRange) * optCumTrapz(diffedF(f0(tRange)) * f(tRange), tRange)