from copy import copy
import numpy as np
import sympy as sp
import pywt
from calculate_eta import eta_k
from phi_basis_functions import getPhiBasisFunctions
from scalar_product import scalarProduct
from optimization import WaveletReconstructedFunction


def ksi_k(Fik, f, F, p, tSpan):
    _eta_k = eta_k(Fik, f, F, p, tSpan)

    ksi = copy(_eta_k)
    # Subtracting projection of eta_k in each phiFunc to get projection of eta_k in W1
    phiBasisFunctions = getPhiBasisFunctions(p, T=tSpan[-1])  # phiFunctions are a basis of W0
    W0coefficients = np.zeros(p)
    for phiIndex, phiFunc in enumerate(phiBasisFunctions):
        # phiFuncInTSpan = phiFunc(tSpan)
        W0coefficients[phiIndex] = scalarProduct(_eta_k, phiFunc, p, tSpan[-1])
        ksi -= W0coefficients[phiIndex] * phiFunc

    # return ksi

    T = tSpan[-1]
    t = sp.Symbol('t')
    # # fourier_series = sp.fourier_series(ksi, (t, 0, T)).truncate(10)
    #
    # wavelet_family = 'db3'
    # levels = int(np.log(len(tSpan)) / np.log(2))
    # time_samples = np.linspace(0, T, 2 ** levels)
    # sampled_ksi = np.array([ksi.subs(t, sample) for sample in time_samples], dtype=np.float64)
    # coeffs = pywt.wavedec(sampled_ksi, wavelet_family, level=levels)
    # reconstructed_function = WaveletReconstructedFunction(coeffs, wavelet_family, levels, T)
    # return reconstructed_function    # T = tSpan[-1]
    # t = sp.Symbol('t')
    # # fourier_series = sp.fourier_series(ksi, (t, 0, T)).truncate(10)
    #
    wavelet_family = 'db3'
    levels = int(np.log(len(tSpan)) / np.log(2))
    time_samples = np.linspace(0, T, 2 ** levels)
    sampled_ksi = np.array([ksi.subs(t, sample) for sample in time_samples], dtype=np.float64)
    coeffs = pywt.wavedec(sampled_ksi, wavelet_family, level=levels)
    reconstructed_function = WaveletReconstructedFunction(coeffs, wavelet_family, levels, T).sympy_implementation()
    return reconstructed_function





    # reconstructed_function = pywt.waverec(coeffs, wavelet_family)
    # plt.figure()
    # plt.plot(tSpan, [ksi.subs(t, tVal) for tVal in tSpan],
    #          label="ksi", lw=2, color='blue')
    # plt.plot(time_samples, reconstructed_function,
    #          label="Approx", lw=1, color='red')
    #
    # plt.legend()
    # plt.show()


def ksiFuncs(Fi, f, F, p, tSpan):
    # p = Degree of derivative used in the smoothness equation
    result = []
    for k, Fik in enumerate(Fi):
        result.append(ksi_k(Fik, f, F, p, tSpan))
        print(f"Calculated ksi_{k}")
    return result
