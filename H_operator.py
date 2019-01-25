from optimization import optCumTrapz,range_memoize

@range_memoize(2)
def Hi(f, F, tRange):
    return optCumTrapz( F( f(tRange) ), tRange, initial=0)

def scalarHi(f, F, tRange):
    return Hi(f, F, tRange)[-1]