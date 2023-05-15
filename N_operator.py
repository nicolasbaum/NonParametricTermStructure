from optimization import quad_with_memoize
import numpy as np
import sympy as sp


def Nk(Fik, f, F, tRangeForBond):
    x, t = sp.symbols('x t')
    integrand = sp.lambdify(t, F.subs(x, f), 'numpy')
    sumands = []
    for i, Fik_i in enumerate(Fik):
        sumands.append(Fik_i * np.exp(-quad_with_memoize(integrand,
                                                         0, tRangeForBond[i])))
    return sum(sumands)
