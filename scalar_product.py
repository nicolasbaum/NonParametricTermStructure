from sympy import Symbol, diff, lambdify, integrate, simplify
from optimization import method_with_sympy_func_memoize, quad_with_memoize, WaveletReconstructedFunction
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz
import numpy as np


# def diffNtimes(myfunction, N):
#     if N == 0:
#         return myfunction
#     if isinstance(myfunction, UnivariateSpline):
#         return myfunction.derivative(N)
#     if isinstance(myfunction, np.ndarray):
#         return np.concatenate([[0], np.diff(myfunction)])
#     x = Symbol('tau')
#     for _ in range(N):
#         f = lambdify(x, diff(myfunction(x), x), "numpy")
#     return f


# def getNormOfFunction(function, tSpan, p):
#     funcInTSpan = function(tSpan)
#     return np.sqrt(scalarProduct(funcInTSpan, funcInTSpan, p, tSpan))


# def scalarProduct(function1, function2, p, tRange):
#     firstSum = 0
#     for i in range(p):  # Will go up to p-1
#         diffedFunction1 = diffNtimes(function1, i)
#         diffedFunction2 = diffNtimes(function2, i)
#         if isinstance(function1, np.ndarray):
#             firstSum += diffedFunction1[0] * diffedFunction2[0]
#         else:
#             firstSum += diffedFunction1(0) * diffedFunction2(0)
#
#     pDiffofFunction1 = diffNtimes(function1, p)
#     pDiffofFunction2 = diffNtimes(function2, p)
#     if isinstance(function1, np.ndarray):
#         pDiffofFunction1EvaluatedInTRange = pDiffofFunction1
#         pDiffofFunction2EvaluatedInTRange = pDiffofFunction2
#     else:
#         pDiffofFunction1EvaluatedInTRange = np.array([pDiffofFunction1(t) for t in tRange])
#         pDiffofFunction2EvaluatedInTRange = np.array([pDiffofFunction2(t) for t in tRange])
#
#     integral = np.trapz(pDiffofFunction1EvaluatedInTRange * pDiffofFunction2EvaluatedInTRange, tRange)
#
#     return firstSum + integral

@method_with_sympy_func_memoize
def scalarProduct(f, g, p, T):
    # if isinstance(f, WaveletReconstructedFunction) and isinstance(g, WaveletReconstructedFunction):
    #     fvector = f.vector()
    #     gvector = g.vector()
    #     fvectorDiff = fvector.copy()
    #     gvectorDiff = gvector.copy()
    #     for i in range(p):
    #         fvectorDiff = (np.concatenate([[0], np.diff(fvectorDiff)]))
    #         gvectorDiff = (np.concatenate([[0], np.diff(gvectorDiff)]))
    #     x = np.linspace(0, T, len(fvector))
    #     return cumtrapz(fvectorDiff * gvectorDiff, x)[-1]

    t = Symbol('t')
    sum_term = sum([f.diff(t, i).subs(t, 0) * g.diff(t, i).subs(t, 0) for i in range(p)])
    integrand = f.diff(t, p) * g.diff(t, p)
    integrand_vectorized = lambdify(t, integrand, 'numpy')
    integral_term = quad_with_memoize(integrand_vectorized, 0, T)
    return sum_term + integral_term
