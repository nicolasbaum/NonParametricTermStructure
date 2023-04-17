from scipy.integrate import cumtrapz
from optimization import range_memoize
import sympy as sp


@range_memoize(2)
def Hi(f, F, tRangeForBond):
    t = sp.Symbol('t')
    return sp.integrate(F(f), (t, 0, tRangeForBond[-1]))


def scalarHi(f, F, tRangeForBond):
    return Hi(f, F, tRangeForBond)[-1]