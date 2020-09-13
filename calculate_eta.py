import numpy as np
from functools import lru_cache,partial
from sympy import Symbol, diff, bell, symbols
from sympy.utilities import lambdify
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz,solve_bvp,solve_ivp
from optimization import range_memoize
from H_operator import Hi

N_POINTS = 1000

def _sthDerivativeOfFunction(s,f):
    x = Symbol('x')
    for _ in range(s):
        if isinstance(f, UnivariateSpline):
            f = f.derivative()
        else:
            f = diff(f(x), x)
    return f


def _sthDerivativeOfCompositeFunctions(s,timeVector, *args):
    if len(args)==2:
        f,g=args #For the time being only 2 functions max
        result=0
        for k in range(1,s+1):
            gDerivatives=[_sthDerivativeOfFunction(i,g) for i in range(1,s-k+2)]
            x=symbols('x:'+str(len(gDerivatives)))
            bellPolynomial=bell(s, k, x)
            lambdifiedBellPoly=lambdify(x,bellPolynomial,"numpy")
            lambdifiedBellPolyEvaluatedInTimeVector,=lambdifiedBellPoly(*zip(gDiffed(timeVector) for gDiffed in gDerivatives))
            result+=_evaluateFunction(_sthDerivativeOfFunction(k,f),g(timeVector))*lambdifiedBellPolyEvaluatedInTimeVector
        return result
    else:
        raise Exception("Invalid number of functions to compose")

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


############################IVP APPROACH############################

def _pDerivativeOfEtaK(t, Fik, f, F, p, tSpan):
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    discountFactors = _discountFactors(f, F, tRangeForBond)
    diffedF = _sthDerivativeOfCompositeFunctions(1, tRangeForBond,F,f)
    integralFunc=np.power(_onlyPositiveSegmentForFunction(lambda u:(t-u),tRangeForBond),(p-1))/np.math.factorial(p-1)
    integral=cumtrapz( diffedF * integralFunc, tRangeForBond, initial=0)

    return -np.sum(discountFactors * Fik[:bondPeriods] * integral)

def _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan):
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    discountFactors = _discountFactors(f, F, tRangeForBond)
    diffedF = _sthDerivativeOfCompositeFunctions(1, tRangeForBond,F,f)
    integralFunc = np.power(tRangeForBond,s)/ np.math.factorial(s)
    integral = cumtrapz(diffedF * integralFunc, tRangeForBond, initial=0)

    return np.sum(discountFactors * Fik[:bondPeriods] * integral)

def _getODEfun(Fik, f, F, p, tSpan):
    func = lambda t:_pDerivativeOfEtaK(t,Fik,f,F,p,tSpan)
    def odefun(func,x,y):
        result=np.zeros(y.shape)
        for i in range(y.shape[0]-1):
            result[i]=y[i+1]
        result[-1]=func(x)
        return result
    return partial(odefun,func)

def _getY0(Fik, f, F, p, tSpan):
    func = lambda s: _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan)
    return [func(s) for s in range(p)]

def _eta_k(Fik, f, F, p, tSpan):
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    result=solve_ivp(_getODEfun(Fik,f,F,p,tRangeForBond),(tRangeForBond[0],tRangeForBond[-1]),_getY0(Fik, f, F, p, tRangeForBond),dense_output=True)
    return (result.t,result.y[0])

# def eta_k(Fik, f, F, p, tSpan):
#     x,y=_eta_k(Fik, f, F, p, tSpan)
#     return UnivariateSpline(x,y)

############################INTEGRAL APPROACH############################

def _inner_sum(t, tau, p):
    inner_sum = np.sum([((t*tau) ** s) / (np.math.factorial(s))**2 for s in range(p)])
    return inner_sum

@lru_cache(maxsize=None)
def _inner_integral(t, tau, p, Tk):
    u=np.linspace(0,Tk,N_POINTS)
    integralFunc=_onlyPositiveSegmentForFunction(lambda x:(t-x),u)*_onlyPositiveSegmentForFunction(lambda x:(tau-x),u)
    return np.trapz( np.power(integralFunc,p-1),u)/(np.math.factorial((p-1))**2)

#@range_memoize(4)
def _outter_integral(F,f,t,p,tRangeForBond):
    diffedF = _sthDerivativeOfCompositeFunctions(1, tRangeForBond,F,f)
    inner_term = np.vectorize( lambda tau: _inner_sum(t, tau, p)+_inner_integral(t,tau,p,tRangeForBond[-1]) )
    return cumtrapz(diffedF*inner_term(tRangeForBond),tRangeForBond, initial=0)

def eta_k(t,Fik, f, F, p, tSpan):
    '''
    :param t: t where I want to evaluate eta_k
    :param Fik: Array of payments of kth bond
    :param f: f function iteration(what we want to solve)
    :param F: function F which transforms f
    :param p: Number of derivative
    :param tSpan: Time vector
    :return: eta for kth bond evaluated in t
    '''
    bondPeriods,tRangeForBond=_bondPeriodsAndTRangeForBond(Fik,tSpan)
    discountFactors=_discountFactors(f, F, tRangeForBond)
    outterIntegral=_outter_integral(F, f, t, p, tRangeForBond)
    return -np.sum( discountFactors  * Fik[:bondPeriods] * outterIntegral )