import numpy as np
from sympy import Symbol, diff
from sympy.utilities import lambdify
from scipy.interpolate import UnivariateSpline
from optimization import optCumTrapz, range_memoize
from H_operator import Hi

N_POINTS = 10000

def _inner_integral(t, tau, p):
    integral_range=min([t,tau])
    u=np.linspace(0,integral_range,N_POINTS)    #ToDo: Validate this...
    return np.trapz( np.power((t-u)*(tau-u),p-1)/(np.math.factorial((p-1))**2),u)

def _sthDerivativeOff(s,f):
    x = Symbol('x')
    for _ in range(s):
        if isinstance(f,UnivariateSpline):
            f = f.derivative()
        else:
            f = diff(f(x), x)
    return f

def _inner_sum(tau, t, p):
    x = Symbol('x')
    inner_sum = np.sum([((t*tau) ** s) / (np.math.factorial(s))**2 for s in range(p)])
    inner_sum + _inner_integral(t, tau, p)
    return inner_sum

def _evaluateFunction(f,points):
    if hasattr(f,'subs'):
        x = Symbol('x')
        fv = lambdify(x,f,"numpy")
        return fv(np.array(points))
    return f(points)

def _outter_integral(F,f,t,p,tRangeForBond):
    x = Symbol('x')
    diffedF = _sthDerivativeOff(1,F)
    inner_sum = np.vectorize( lambda tau: _inner_sum(tau,t,p) )
    return optCumTrapz(_evaluateFunction(diffedF,f(tRangeForBond))*inner_sum(tRangeForBond),tRangeForBond, initial=0)

def eta_k(t,Fik,f,F,p,tRange):
    '''
    :param t: t where I want to evaluate eta_k
    :param Fik: Array of payments of kth bond
    :param f: f function iteration(what we want to solve)
    :param F: function F which transforms f
    :param p: Number of derivative
    :param tRange: Time vector
    :return: eta for kth bond evaluated in t
    '''
    bondPeriods = int(np.nonzero(Fik)[0][-1])+1
    tRangeForBond = tRange[:bondPeriods]
    return -np.sum(  np.exp( -Hi(f, F, tRangeForBond) ) * Fik[:bondPeriods] * _outter_integral(F, f, t, p, tRangeForBond) )