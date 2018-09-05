import numpy as np
from sympy import Symbol, diff, lambdify
from N_operator import Nk

N_POINTS = 10000

def _inner_integral(t, tau, p):
    integral_range=min(t,tau)
    u=np.linspace(0,integral_range,N_POINTS)    #ToDo: Validate this...
    return np.trapz( np.power((t-u)*(tau-u),p-1)/(np.factorial((p-1))**2),u)

def _sthDerivativeOff(s,f):
    x = Symbol('x')
    for _ in range(s):
        f = lambdify(x, diff(f(x), x), "numpy")
    return f

def _inner_sum(tau,f, t, p):
    inner_sum = np.sum([(t ** s) * _sthDerivativeOff(s, f)(0) / np.math.factorial(s) for s in range(p)])
    inner_sum + _inner_integral(t, tau, p)

def _outter_integral(F,f,t,p,tRange):
    x = Symbol('tau')
    diffedF = lambdify(x, diff(F(x), x), "numpy")
    inner_sum = lambdify(x, _inner_sum(x,f,t,p))
    return np.trapz(diffedF(tRange)*inner_sum(tRange),tRange)

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

    return -Nk(Fik, f, F, tRange) * _outter_integral(F, f, t, p, tRange)