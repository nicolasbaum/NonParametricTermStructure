import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sympy import Symbol, diff, lambdify


def sthDerivativeOff(s, f):
    x = Symbol('x')
    for _ in range(s):
        if isinstance(f,InterpolatedUnivariateSpline):
            f = f.derivative()
        elif isinstance(f,np.ndarray):
            f=np.concatenate(([0],np.diff(f)))
        else:
            f = diff(f(x), x)
    return f


def evaluateFunction(f, points):
    if hasattr(f,'subs'):
        x = Symbol('x')
        fv = lambdify(x,f,"numpy")
        return fv(np.array(points))
    return f(points)