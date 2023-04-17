from calculate_eta import eta_k
import numpy as np
from phi_basis_functions import getPhiBasisFunctions
from copy import copy
from scalar_product import scalarProduct


def ksi_k(Fik, f, F, p, tSpan):
    phiBasisFunctions = getPhiBasisFunctions(p, T=tSpan[-1])  # phiFunctions are a basis of W0
    W0coefficients = np.zeros(p)

    _eta_k = eta_k(Fik, f, F, p, tSpan)
    # eta_kEvaluatedInTSpan = eta_kVectorized(tSpan)
    # from matplotlib import pyplot as plt
    # plt.plot(tSpan,eta_kEvaluatedInTSpan)
    ksi_k = copy(_eta_k)
    # Subtracting projection of eta_k in each phiFunc to get projection of eta_k in W1
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        # phiFuncInTSpan = phiFunc(tSpan)
        W0coefficients[phiIndex] = scalarProduct(_eta_k, phiFunc, p, tSpan[-1])
        ksi_k -= W0coefficients[phiIndex] * phiFunc

    return ksi_k


def ksiFuncs(Fi, f, F, p, tSpan):
    # p = Degree of derivative used in the smoothness equation

    #COMMENTED LINES FOR DEBUG
    from matplotlib import pyplot as plt
    plt.figure()
    a = [ksi_k(Fik, f, F, p, tSpan)(tSpan) for Fik in Fi]
    plt.draw()
    return a
