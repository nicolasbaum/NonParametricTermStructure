from optimization import optCumTrapz
from sympy import Symbol, diff, lambdify
from H_operator import Hi
import numpy as np

def Dk(Fik, f0, f, F, tRangeForBond):
    x = Symbol('x')
    #Revisar esto de la diferenciacion de F
    diffedF = lambdify( x, diff( F(x), x ), "numpy" )
    return -np.sum( np.exp(-Hi(f0,F,tRangeForBond)) *Fik* optCumTrapz(diffedF(f0(tRangeForBond)) * f(tRangeForBond), tRangeForBond, initial=0) )