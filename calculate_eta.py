import numpy as np
from functools import lru_cache,partial
from sympy import Symbol, diff
from sympy.utilities import lambdify
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz,solve_bvp
from optimization import range_memoize
from H_operator import Hi

N_POINTS = 1000

def _sthDerivativeOff(s,f):
    x = Symbol('x')
    for _ in range(s):
        if isinstance(f,UnivariateSpline):
            f = f.derivative()
        else:
            f = diff(f(x), x)
    return f

def _evaluateFunction(f,points):
    if hasattr(f,'subs'):
        x = Symbol('x')
        fv = lambdify(x,f,"numpy")
        return fv(np.array(points))
    return f(points)

def _discountFactors(f, F, tRangeForBond):
    return np.exp( -Hi(f, F, tRangeForBond) )

def _bondPeriodsAndTRangeForBond(Fik,tSpan):
    bondPeriods = int(np.nonzero(Fik)[0][-1]) + 1
    return (bondPeriods,tSpan[:bondPeriods])

def _onlyPositiveSegmentForFunction(func,x):
    result=_evaluateFunction(func, x)
    return np.clip(result,0,None)


############################BVP4C APPROACH############################

def _pDerivativeOfEtaK(t, Fik, f, F, p, tSpan):
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    discountFactors = _discountFactors(f, F, tRangeForBond)
    diffedF = _evaluateFunction(_sthDerivativeOff(1, F), f(tRangeForBond))
    integralFunc=np.power(_onlyPositiveSegmentForFunction(lambda u:(t-u),tRangeForBond),(p-1))/np.math.factorial(p-1)
    integral=cumtrapz( diffedF * integralFunc, tRangeForBond, initial=0)

    return -np.sum(discountFactors * Fik[:bondPeriods] * integral)

def _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan):
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    discountFactors = _discountFactors(f, F, tRangeForBond)
    diffedF = _evaluateFunction(_sthDerivativeOff(1, F), f(tRangeForBond))
    integralFunc = np.power(tRangeForBond,s)/ np.math.factorial(s)
    integral = cumtrapz(diffedF * integralFunc, tRangeForBond, initial=0)

    return np.sum(discountFactors * Fik[:bondPeriods] * integral)

def _getODEfun(Fik, f, F, p, tSpan):
    func = lambda t:_pDerivativeOfEtaK(t,Fik,f,F,p,tSpan)
    def odefun(func,x,y):
        result=np.zeros(y.shape)
        for i in range(y.shape[0]-1):
            result[i,:]=y[i+1]
        result[-1,:]=[func(xi) for xi in x]
        return result
    return partial(odefun,func)

def _getBCfun(Fik, f, F, p, tSpan):
    func = lambda s: _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan)
    def bcfun(func,ya,yb):
        result=[ya[i]-func(i) for i in range(ya.shape[0])]
        return result
    return partial(bcfun,func)

def _initialGuess(p,tSpan):
    indexes = list(range(p-1))
    funcs=[]
    sin=np.sin(tSpan)
    cos=np.cos(tSpan)
    for i in indexes:
        funcs.append(sin if i%2 else cos)
    return np.vstack(funcs)

#@range_memoize(4)
def _eta_k(Fik, f, F, p, tSpan):
    result=solve_bvp(_getODEfun(Fik,f,F,p,tSpan),_getBCfun(Fik,f,F,p,tSpan),tSpan,_initialGuess(p,tSpan))
    return result.sol

def eta_k(Fik, f, F, p, tSpan):
    return _eta_k(Fik, f, F, p, tSpan)
############################INTEGRAL APPROACH############################

def _inner_sum(t, tau, p):
    inner_sum = np.sum([((t*tau) ** s) / (np.math.factorial(s))**2 for s in range(p)])
    return inner_sum

@lru_cache(maxsize=None)
def _inner_integral(t, tau, p, Tk):
    u=np.linspace(0,Tk,N_POINTS)
    integralFunc=_onlyPositiveSegmentForFunction(lambda x:(t-x),u)*_onlyPositiveSegmentForFunction(lambda x:(tau-x),u)
    return np.trapz( np.power(integralFunc,p-1),u)/(np.math.factorial((p-1))**2)

@range_memoize(4)
def _outter_integral(F,f,t,p,tRangeForBond):
    diffedF = _sthDerivativeOff(1,F)
    inner_term = np.vectorize( lambda tau: _inner_sum(t, tau, p)+_inner_integral(t,tau,p,tRangeForBond[-1]) )
    return cumtrapz(_evaluateFunction(diffedF,f(tRangeForBond))*inner_term(tRangeForBond),tRangeForBond, initial=0)

# def eta_k(t, Fik, f, F, p, tSpan):
#     '''
#     :param t: t where I want to evaluate eta_k
#     :param Fik: Array of payments of kth bond
#     :param f: f function iteration(what we want to solve)
#     :param F: function F which transforms f
#     :param p: Number of derivative
#     :param tSpan: Time vector
#     :return: eta for kth bond evaluated in t
#     '''
#     bondPeriods,tRangeForBond=_bondPeriodsAndTRangeForBond(Fik,tSpan)
#     discountFactors=_discountFactors(f, F, tRangeForBond)
#     outterIntegral=_outter_integral(F, f, t, p, tRangeForBond)
#
#     return -np.sum( discountFactors  * Fik[:bondPeriods] * outterIntegral )