from scipy.integrate import cumtrapz
from optimization import range_memoize

#@range_memoize(2)
def Hi(f, F, tRangeForBond):
    return cumtrapz( F( f(tRangeForBond) ), tRangeForBond, initial=0)

def scalarHi(f, F, tRangeForBond):
    return Hi(f, F, tRangeForBond)[-1]