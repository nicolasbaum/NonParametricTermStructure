import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator, NelsonSiegelModel
from optimization import PolynomialModel
from tilde_y import tilde_y
from calculate_T import getT
from calculate_M import getM
from calculate_ksi import ksiFuncs
from calculate_r import zVectorFromf
from phi_basis_functions import getPhiBasisFunctions
from copy import deepcopy
import datetime


bcl = BondCalendarLoader('calendars.xlsx')
cl = bcl.getCalendarList()
bondPrices = bcl.getBondPrices()
P = bondPrices['Price'].values
Fi = cl.getPaymentMatrix()

ytmCalc = YTMCalculator(calcDate=datetime.date(2018,9,12), yearConvention=365.25)
yieldCurve = ytmCalc.getInterpolatedYieldCurve(cl.calendars, bondPrices)

macaulyDurations,_,_ = ytmCalc.getDurationsYTMsAndMaturities(cl.calendars, bondPrices)
scaleFactors = 1/macaulyDurations
for i,sf in enumerate(scaleFactors):
    P[i] *= sf
    Fi[i,:] *= sf

N = Fi.shape[0]
tSpan = np.arange(1,Fi.shape[1]+1)/12.0

x = sp.Symbol('x')
F = x ** 2
invF = sp.sqrt(x)

Lambda = 0.6
p = 2     # derivative degree used in smoothness equation

"""Since Spot curve from yield curve requires some form of bootstrap
and this isn't easy because settlement dates are not overlapping,
I'm going to use as a (bad) proxy, the yields as the spots.
"""
r0 = yieldCurve(tSpan)
t = sp.Symbol('t')
# f0 = 1-sp.exp(-t)
# f0 = NelsonSiegelModel(tSpan, r0).sympy_implementation()
f0 = PolynomialModel(tSpan, r0).sympy_implementation()
f0Basis = getPhiBasisFunctions(p+1, tSpan[-1])[1:]

z0 = zVectorFromf(f0,F,tSpan)

steps = 0
while steps < 10:
    print(f"Iteration #{steps}")
    # Idea behind f=f0 is that I want to converge to the final result where previous iteration = current f
    print("Calculating ksi functions")
    ksi_functions = ksiFuncs(Fi, f0, F, p, tSpan)
    print("Calculating M")
    M = getM(Lambda,Fi,ksi_functions,p,tSpan)
    print("Calculating T")
    T = getT(p, Fi, f0, F, tSpan)
    print("Calculating tilde y")
    y = tilde_y(P, Fi, f0, tSpan,F)

    # Check original idea that uses QR decomposition
    invM = np.linalg.inv(np.asmatrix(M))
    aux = np.linalg.inv(T.transpose()@invM@T)@T.transpose()@invM
    aux2 = T @ aux
    d = np.squeeze(np.array(aux@y))
    c = np.squeeze(np.array(invM @ (np.eye(len(aux2)) - aux2) @ y))

    previous_f = deepcopy(f0)

    phiSum = sum(d[basisIndex]*f0BasisFunc for basisIndex, f0BasisFunc in enumerate(f0Basis))
    ksiSum = sum(c[ksiIndex]*ksiFunc for ksiIndex, ksiFunc in enumerate(ksi_functions))
    print("Simplifying f0")
    f0 = sp.simplify(phiSum+ksiSum)

    # z0Calculated = zVectorFromf(f0,F,tSpan)
    plt.plot(tSpan, sp.lambdify(t, f0,'numpy')(tSpan), label=f"Iteration {steps}")

    steps += 1

plt.plot(tSpan, r0, 'b', label='r0')
plt.legend()
plt.show()
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import cm
# color=iter(cm.rainbow(np.linspace(0,1,len(f0Basis))))
#
# for phiFunc in f0Basis:
#     phi = phiFunc(tSpan)/getNormOfFunction(phiFunc, tSpan, p)
#     plt.plot(tSpan, phi, c=next(color))
