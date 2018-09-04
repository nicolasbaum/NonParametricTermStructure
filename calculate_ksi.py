from calculate_eta import eta_k
import numpy as np
from sympy import Symbol, lambdify
from phi_basis_functions import getPhiBasisFunctions

def ksi_k(Fik, f, F, p, tRange):
    phiBasisFunctions = getPhiBasisFunctions(p)
    ksi=np.zeros(p)

    t = Symbol('t')
    eta_kVectorized = lambdify(t, eta_k(t, Fik, f, F, p, tRange), "numpy")
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        ksi[phiIndex]=np.trapz( eta_kVectorized(tRange)*phiFunc(tRange), tRange )

    return ksi