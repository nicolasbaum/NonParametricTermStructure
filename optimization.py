import numpy as np
from numba import jit
import scipy.integrate
from scipy.interpolate.interpolate import interp1d

optCumTrapz = jit(scipy.integrate.cumtrapz, parallel = True, fastmath=True )
optTrapz = jit(scipy.integrate.trapz, parallel = True, fastmath=True)

class Interp1dNumeric(interp1d):
    """ Wrapper for interp1 to raise TypeError for object array input
    We need this because sympy will try to evaluate interpolated functions when
    constructing expressions involving floats.  At least sympy 1.0 only accepts
    TypeError or AttributeError as indication that the implemented value cannot
    be sampled with the sympy expression.  Therefore, raise a TypeError
    directly for an input giving an object array (such as a sympy expression),
    rather than letting interp1d raise a ValueError.
    See:
    * https://github.com/nipy/nipy/issues/395
    * https://github.com/sympy/sympy/issues/10810
    """
    def __call__(self, x):
        if np.asarray(x).dtype.type == np.object_:
            raise TypeError('Object arrays not supported')
        return super(Interp1dNumeric, self).__call__(x)

