from scipy.integrate import cumtrapz
from optimization import optCumTrapz

def Hi(f, F, tRange):
    return optCumTrapz( F( f(tRange) ), tRange, initial=0)