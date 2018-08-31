import numpy as np
from sympy import Symbol, diff, lambdify
from N_operator import Nk

N_POINTS = 10000

def inner_integral(t,tau,p,T):
    '''
    :param t: point where eta is trying to be calculated
    :param tau: integral variable for the outter integral
    :param p: number of derivative
    :param T: max(ti), where ti is the time where current payment Fi is being payed
              Fik: Payment at time i of kth bond
    '''
    integral_range=min(t,tau,T)
    u=np.linspace(0,integral_range,N_POINTS)
    return np.trapz( np.power((t-u)*(tau-u),p-1)/(np.factorial((p-1))**2),u)

def _sthDerivativeOff(s,f):
    x = Symbol('x')
    for _ in range(s):
        f = lambdify(x, diff(f(x), x), "numpy")
    return f

def _inner_sum(tau,f, t, p, T):
    inner_sum = np.sum([(t ** s) * _sthDerivativeOff(s, f)(0) / np.math.factorial(s) for s in range(p)])
    inner_sum + inner_integral(t, tau, p, T)

def _outter_integral(F,f,t,ti,p,T,dt):
    tRange = np.linspace(0, ti, int(ti / dt))

    x = Symbol('tau')
    diffedF = lambdify(x, diff(F(x), x), "numpy")
    inner_sum = lambdify(x, _inner_sum(x,f,t,p,T))
    return np.trapz(diffedF(tRange)*inner_sum(tRange),tRange)

def eta_k(t,Fi,f,F,p):
    '''
    :param t: t where I want to evaluate eta_k
    :param Fi: Array of payments of kth bond
    :param f: f function iteration(what we want to solve)
    :param F: function F which transforms f
    :param p: Number of derivative
    :return: eta for kth bond evaluated in t
    '''

    #Fi should be enough to infer ti, dt and T
    #ToDo: Finish implementing this

    return -Nk(Fi,f,F,ti)*_outter_integral(F,f,t,ti,p,T,dt)