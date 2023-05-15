from optimization import range_memoize
from optimization import quad_with_memoize
import sympy as sp


@range_memoize(2)
def Hi(f, F, tRangeForBond):
    t, x = sp.symbols('t x')
    lambdified_integrand = sp.lambdify(t, F.subs(x, f), 'numpy')
    return quad_with_memoize(lambdified_integrand, 0, tRangeForBond[-1])
