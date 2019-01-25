from calculate_eta import eta_k
import numpy as np
from phi_basis_functions import getPhiBasisFunctions
from scipy.interpolate import UnivariateSpline

def ksi_k(Fik, f, F, p, tRange):
    phiBasisFunctions = getPhiBasisFunctions(p, tRange)     #phiFunctions are a basis of W0
    W0coefficients=np.zeros(p)

    eta_kVectorized = np.vectorize(lambda x: eta_k(x, Fik, f, F, p, tRange))
    eta_kEvaluatedInTRange = eta_kVectorized(tRange)
    ksi_kEvaluatedInTRange = eta_kEvaluatedInTRange
    #Subtracting projection of eta_k in each phiFunc to get projection of eta_k in W1
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        phiFuncInTRange = phiFunc(tRange)
        W0coefficients[phiIndex]=np.trapz( eta_kEvaluatedInTRange*phiFuncInTRange, tRange )
        ksi_kEvaluatedInTRange -= W0coefficients[phiIndex]*phiFuncInTRange

    return UnivariateSpline(tRange, ksi_kEvaluatedInTRange)