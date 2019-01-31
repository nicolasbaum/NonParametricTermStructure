from calculate_eta import eta_k
import numpy as np
from phi_basis_functions import getPhiBasisFunctions
from scipy.interpolate import UnivariateSpline
from scalar_product import scalarProduct, getNormOfFunction


def ksi_k(Fik, f, F, p, tSpan):
    phiBasisFunctions = getPhiBasisFunctions(p)  # phiFunctions are a basis of W0
    W0coefficients = np.zeros(p)

    eta_kVectorized = np.vectorize(lambda x: eta_k(x, Fik, f, F, p, tSpan))
    eta_kEvaluatedInTSpan = eta_kVectorized(tSpan)
    ksi_kEvaluatedInTSpan = eta_kEvaluatedInTSpan.copy()
    # Subtracting projection of eta_k in each phiFunc to get projection of eta_k in W1
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        phiFuncInTSpan = phiFunc(tSpan)
        W0coefficients[phiIndex] = scalarProduct(eta_kEvaluatedInTSpan, phiFuncInTSpan, p, tSpan) \
                                   / getNormOfFunction(phiFunc, tSpan, p)
        ksi_kEvaluatedInTSpan -= W0coefficients[phiIndex] * phiFuncInTSpan

    return UnivariateSpline(tSpan, ksi_kEvaluatedInTSpan)


def ksiFuncs(Fi, f, F, p, tSpan):
    # p = Degree of derivative used in the smoothness equation
    return [ksi_k(Fik, f, F, p, tSpan) for Fik in Fi]
