from optimization import quad_with_memoize
import sympy as sp
import numpy as np


def Dk(Fik, f0, f, F, tRangeForBond):
    x, t = sp.symbols('x t')
    inner_integrand_vectorized = sp.lambdify(t, f * F.diff().subs(x, f0), 'numpy')
    outer_integrand = sp.lambdify(t, F.subs(x, f0), 'numpy')
    sumands = []
    for i, Fik_i in enumerate(Fik):
        sumands.append(np.exp(-quad_with_memoize(outer_integrand,
                                                 0, tRangeForBond[i])) * Fik_i *
                       quad_with_memoize(inner_integrand_vectorized,
                                         0, tRangeForBond[i]))
    return -sum(sumands)
