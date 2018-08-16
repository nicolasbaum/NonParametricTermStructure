import numpy as np
from matplotlib import pyplot as plt
from N_operator import Nk
from sympy import Symbol, diff, lambdify

def Dk( Fi, f0, f, F=None, ti=1, dt=0.001 ):
    x = Symbol('x')
    tRange = np.linspace(0,ti,int(ti/dt))
    #Revisar esto de la diferenciacion de F
    diffedF = lambdify( x, diff( F(x), x ), "numpy" )
    return -1*Nk(Fi,f0,F,ti,dt)*np.trapz(diffedF(f0(tRange))*f(tRange))