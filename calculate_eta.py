import numpy as np
from functools import lru_cache
from scipy.integrate import cumtrapz
import sympy as sp

from algebra import sthDerivativeOff, evaluateFunction
from optimization import range_memoize
from H_operator import Hi

N_POINTS = 1000


def _bondPeriodsAndTRangeForBond(Fik, tSpan):
    bondPeriods = int(np.nonzero(Fik)[0][-1]) + 1
    return bondPeriods, tSpan[:bondPeriods]


def eta_k(Fik, f, F, p, tSpan):
    """
    :param t: t where I want to evaluate eta_k
    :param Fik: Array of payments of kth bond
    :param f: f function iteration(what we want to solve)
    :param F: function F which transforms f
    :param p: Number of derivative
    :param tSpan: Time vector
    :return: eta for kth bond evaluated in t
    """
    bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
    T = tSpan[-1]
    # Define symbols
    t, tau, u, ti, x = sp.symbols('t tau u ti x')
    n, i, s = sp.symbols('n i s', integer=True)

    ti_piecewise = sp.Piecewise(*[(tRangeForBond[idx], x <= idx) for idx in range(bondPeriods)]).subs(x, i)
    Fik_piecewise = sp.Piecewise(*[(Fik[idx], x <= idx) for idx in range(bondPeriods)]).subs(x, i)

    # Define the exponential term
    exponential_term = sp.exp(-sp.integrate(F.subs(x, f), (tau, ti))) * Fik_piecewise
    exponential_term = exponential_term.subs(ti, ti_piecewise)

    # Define the integral term
    taylor_term = sp.Sum((t ** s) * (tau ** s) / (sp.factorial(s)) ** 2, (s, 0, p - 1))
    integral_reminder_term = sp.integrate((t - u) ** (p - 1) * (tau - u) ** (p - 1) / ((sp.factorial(p - 1)) ** 2),
                                          (u, 0, T))

    taylor_term = sp.simplify(taylor_term)
    integral_reminder_term = sp.simplify(integral_reminder_term)
    integral_term = sp.integrate(F.diff().subs(x, f) * (taylor_term + integral_reminder_term), (tau, ti)).subs(ti, ti_piecewise)

    # Combine the terms to find eta_k(t)
    _eta_k = -sum(exponential_term.subs(i, period) * integral_term.subs(i, period) for period in range(bondPeriods))

    return _eta_k

    # Simplify the expression
    #eta_k_simplified = sp.simplify(_eta_k)
    # return eta_k_simplified


############################IVP APPROACH############################

# def _pDerivativeOfEtaK(t, Fik, f, F, p, tSpan):
#     bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
#     discountFactors = _discountFactors(f, F, tRangeForBond)
#     diffedF = _evaluateFunction(_sthDerivativeOff(1, F), f(tRangeForBond))
#     integralFunc=np.power(_onlyPositiveSegmentForFunction(lambda u:(t-u),tRangeForBond),(p-1))/np.math.factorial(p-1)
#     integral=cumtrapz( diffedF * integralFunc, tRangeForBond, initial=0)
#
#     return -np.sum(discountFactors * Fik[:bondPeriods] * integral)

# def _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan):
#     bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
#     discountFactors = _discountFactors(f, F, tRangeForBond)
#     diffedF = _evaluateFunction(_sthDerivativeOff(1, F), f(tRangeForBond))
#     integralFunc = np.power(tRangeForBond,s)/ np.math.factorial(s)
#     integral = cumtrapz(diffedF * integralFunc, tRangeForBond, initial=0)
#
#     return np.sum(discountFactors * Fik[:bondPeriods] * integral)

# def _getODEfun(Fik, f, F, p, tSpan):
#     func = lambda t:_pDerivativeOfEtaK(t,Fik,f,F,p,tSpan)
#     def odefun(func,x,y):
#         result=np.zeros(y.shape)
#         for i in range(y.shape[0]-1):
#             result[i]=y[i+1]
#         result[-1]=func(x)
#         return result
#     return partial(odefun,func)

# def _getY0(Fik, f, F, p, tSpan):
#     func = lambda s: _sDerivativeOfEtaKIn0(Fik, f, F, s, tSpan)
#     return [func(s) for s in range(p)]

# def _eta_k(Fik, f, F, p, tSpan):
#     result=solve_ivp(_getODEfun(Fik,f,F,p,tSpan),(tSpan[0],tSpan[-1]),_getY0(Fik, f, F, p, tSpan),dense_output=True)
#     return (result.t,result.y[0])

# def eta_k(Fik, f, F, p, tSpan):
#     x,y=_eta_k(Fik, f, F, p, tSpan)
#     return UnivariateSpline(x,y)





############################INTEGRAL APPROACH############################
# def _discountFactors(f, F, tRangeForBond):
#     return np.exp(-Hi(f, F, tRangeForBond))
#
#
#
#
# def _onlyPositiveSegmentForFunction(func, x):
#     result = evaluateFunction(func, x)
#     return np.clip(result, 0, None)
#
# def _inner_sum(t, tau, p):
#     inner_sum = np.sum([((t * tau) ** s) / (np.math.factorial(s)) ** 2 for s in range(p)])
#     return inner_sum
#
#
# @lru_cache(maxsize=None)
# def _inner_integral(t, tau, p, Tk):
#     u = np.linspace(0, Tk, N_POINTS)
#     integralFunc = _onlyPositiveSegmentForFunction(lambda x: (t - x), u) * _onlyPositiveSegmentForFunction(
#         lambda x: (tau - x), u)
#     return np.trapz(np.power(integralFunc, p - 1), u) / (np.math.factorial((p - 1)) ** 2)
#
#
# @range_memoize(4)
# def _outter_integral(F, f, t, p, tRangeForBond, tSpan):
#     diffedF = sthDerivativeOff(1, F)
#     inner_term = np.vectorize(lambda tau: _inner_sum(t, tau, p) + _inner_integral(t, tau, p, tSpan[-1]))
#     return cumtrapz(evaluateFunction(diffedF, f(tRangeForBond)) * inner_term(tRangeForBond), tRangeForBond, initial=0)
#
#
# def eta_k(t, Fik, f, F, p, tSpan):
#     """
#     :param t: t where I want to evaluate eta_k
#     :param Fik: Array of payments of kth bond
#     :param f: f function iteration(what we want to solve)
#     :param F: function F which transforms f
#     :param p: Number of derivative
#     :param tSpan: Time vector
#     :return: eta for kth bond evaluated in t
#     """
#     bondPeriods, tRangeForBond = _bondPeriodsAndTRangeForBond(Fik, tSpan)
#     discountFactors = _discountFactors(f, F, tRangeForBond)
#     outerIntegral = _outter_integral(F, f, t, p, tRangeForBond, tSpan)
#     return -np.sum(discountFactors * Fik[:bondPeriods] * outerIntegral)
#
