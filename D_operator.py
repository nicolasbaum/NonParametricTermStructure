# D_operator.py

from scipy.integrate import cumtrapz
from sympy import Symbol, diff, lambdify
from H_operator import Hi
import numpy as np

def Dk(Fik, f0, directionFunc, F, tRangeForBond):
    """
    For bond's row Fik, partial derivative ~
     - sum_{i} [ e^{-Hi(f0)} * Fik[i] * \int_{0}^{t_i} F'( f0(t) ) * directionFunc(t) dt ]
    implemented discretely via cumtrapz.
    """
    # 1) Make diffF = F'(x):
    x = Symbol('x')
    diffedF = lambdify(x, diff(F(x), x), "numpy")

    # 2) Evaluate e^{-H_i(f0)}
    discount = np.exp(-Hi(f0, F, tRangeForBond))

    # 3) Evaluate integrals of F'(f0(t)) * directionFunc(t) up to each point
    #    So inside cumtrapz we pass diffedF(f0(tRangeForBond)) * directionFunc(tRangeForBond).
    integrals = cumtrapz(
        diffedF(f0(tRangeForBond)) * directionFunc(tRangeForBond),
        tRangeForBond, 
        initial=0
    )
    # 4) Multiply each i-th bond payment with discount_i times integrals_i,
    #    sum up, then put a negative sign
    return -np.sum(discount * Fik * integrals)