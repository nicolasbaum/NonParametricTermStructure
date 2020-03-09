import numpy as np
from functools import lru_cache
from sympy import Symbol, diff
from sympy.utilities import lambdify
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz
from optimization import range_memoize
from H_operator import Hi

N_POINTS = 1000

@lru_cache(maxsize=None)
#@jit
def _inner_integral(t, tau, p):
    integral_range=min([t,tau])
    u=np.linspace(0,integral_range,N_POINTS)    #ToDo: Validate this...
    return np.trapz( np.power((t-u)*(tau-u),p-1),u)/(np.math.factorial((p-1))**2)

def _sthDerivativeOff(s,f):
    x = Symbol('x')
    for _ in range(s):
        if isinstance(f,UnivariateSpline):
            f = f.derivative()
        else:
            f = diff(f(x), x)
    return f

def _inner_sum(t, tau, p):
    x = Symbol('x')
    inner_sum = np.sum([((t*tau) ** s) / (np.math.factorial(s))**2 for s in range(p)])
    return inner_sum

def _evaluateFunction(f,points):
    if hasattr(f,'subs'):
        x = Symbol('x')
        fv = lambdify(x,f,"numpy")
        return fv(np.array(points))
    return f(points)

#@range_memoize(4)
def _outter_integral(F,f,t,p,tRangeForBond):
    x = Symbol('x')
    diffedF = _sthDerivativeOff(1,F)
    inner_term = np.vectorize( lambda tau: _inner_sum(t, tau, p)+_inner_integral(t, tau, p) )
    return cumtrapz(_evaluateFunction(diffedF,f(tRangeForBond))*inner_term(tRangeForBond),tRangeForBond, initial=0)

def eta_k(t, Fik, f, F, p, tSpan):
    '''
    :param t: t where I want to evaluate eta_k
    :param Fik: Array of payments of kth bond
    :param f: f function iteration(what we want to solve)
    :param F: function F which transforms f
    :param p: Number of derivative
    :param tSpan: Time vector
    :return: eta for kth bond evaluated in t
    '''
    bondPeriods = int(np.nonzero(Fik)[0][-1])+1
    tRangeForBond = tSpan[:bondPeriods]

    discountedCasflows=np.exp( -Hi(f, F, tRangeForBond) )
    outterIntegral=_outter_integral(F, f, t, p, tRangeForBond)

    return -np.sum( discountedCasflows  * Fik[:bondPeriods] * outterIntegral )