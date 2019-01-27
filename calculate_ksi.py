from calculate_eta import eta_k
import numpy as np
from phi_basis_functions import getPhiBasisFunctions
from scipy.interpolate import UnivariateSpline
from scalar_product import scalarProduct

def ksi_k(Fik, f, F, p, tRangeForBond):
    phiBasisFunctions = getPhiBasisFunctions(p)     #phiFunctions are a basis of W0
    W0coefficients=np.zeros(p)

    eta_kVectorized = np.vectorize(lambda x: eta_k(x, Fik, f, F, p, tRangeForBond))
    eta_kEvaluatedInTRange = eta_kVectorized(tRangeForBond)
    ksi_kEvaluatedInTRange = eta_kEvaluatedInTRange.copy()
    #Subtracting projection of eta_k in each phiFunc to get projection of eta_k in W1
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        phiFuncInTRange = phiFunc(tRangeForBond)
        W0coefficients[phiIndex]= scalarProduct(eta_kEvaluatedInTRange, phiFuncInTRange, p, tRangeForBond) / np.sqrt(scalarProduct(phiFuncInTRange, phiFuncInTRange, p, tRangeForBond))
        ksi_kEvaluatedInTRange -= W0coefficients[phiIndex]*phiFuncInTRange

    return UnivariateSpline(tRangeForBond, ksi_kEvaluatedInTRange)

def ksiFuncs(Fi, f, F, p, tRangeForBond):
    #p = Degree of derivative used in the smoothness equation
    return [ksi_k(Fik, f, F, p, tRangeForBond) for Fik in Fi]