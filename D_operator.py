from optimization import optCumTrapz
from sympy import Symbol, diff, lambdify
from H_operator import Hi
import numpy as np

def Dk(Fik, f0, f, F, tRange):
    x = Symbol('x')
    #Revisar esto de la diferenciacion de F
    diffedF = lambdify( x, diff( F(x), x ), "numpy" )
    return -np.sum( np.exp(-Hi(f0,F,tRange)) *Fik* optCumTrapz(diffedF(f0(tRange)) * f(tRange), tRange, initial=0) )