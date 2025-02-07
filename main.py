import numpy as np
from sympy import Symbol, lambdify
from scipy.interpolate import interp1d
import datetime
from copy import deepcopy

# Local imports from the same directory:
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator, bondYTMfromZeroDiscount
from tilde_y import tilde_y   # must be updated to accept 'direction'
from calculate_T import getT
from calculate_M import getM
from calculate_ksi import ksiFuncs
from calculate_r import zVectorFromf, dVectorFromf
from phi_basis_functions import getPhiBasisFunctions
from scalar_product import getNormOfFunction
from algebra import sthDerivativeOff

############################################################
# 1. Load data from 'calendars.xlsx'
############################################################

bcl = BondCalendarLoader('calendars.xlsx')
cl = bcl.getCalendarList()
bondPrices = bcl.getBondPrices()

# P is a vector of bond prices, Fi is the matrix of bond cashflows
P = bondPrices['Price'].values  # shape = (#bonds,)
Fi = cl.getPaymentMatrix()      # shape = (#bonds, #time_points)

############################################################
# 2. Possibly rescale prices by Macaulay durations or do
#    whatever the original code does.
############################################################

ytmCalc = YTMCalculator(calcDate=datetime.date(2018,9,12), yearConvention=365.25)
# Example: you might scale each bond by 1/duration. This was in the original code.
# Macaulay durations:
macaulyDurations, yields, maturities = ytmCalc.getDurationsYTMsAndMaturities(cl.calendars, bondPrices)
scaleFactors = 1.0 / macaulyDurations

for i, sf in enumerate(scaleFactors):
    P[i]  *= sf
    Fi[i,:] *= sf

############################################################
# 3. Setup time grid and the function F(x) = x^2
############################################################

N = Fi.shape[0]  # number of bonds
tSpan = np.arange(1, Fi.shape[1] + 1) / 12.0   # e.g. 1..m months in steps of 1/12

x = Symbol('x', real=True, nonnegative=True)
F_expr = x**2  # if the method is F(x)=x^2
F = lambdify(x, F_expr, "numpy")

# The derivative will be needed in D_operator, so ensure D_operator also uses:
#   diffedF = lambdify(x, diff(F_expr, x), "numpy")

############################################################
# 4. Build initial guess f0 from a naive approach
#    We'll do something like: r0(t) = 0.08 constant, then f0(t) = sqrt(r0).
############################################################

# For example, let’s do a constant yield = 8% on tSpan:
r0 = 0.08 * np.ones_like(tSpan)  # shape = (#time_points,)

# f0(t) such that F( f0 ) = (f0)^2 = r(t). So f0 = sqrt( r0(t) ).
f0Vector = np.sqrt(r0)  # shape = (#time_points,)

# We want f0 to be an interpolating function. Use e.g. 'cubic' or 'linear'.
f0 = interp1d(tSpan, f0Vector, kind='linear', fill_value="extrapolate")

############################################################
# 5. Choose derivative order p and a smoothing param Lambda
#    (You can implement GCV or GML if you want auto-lambda.)
############################################################

p = 1
Lambda = 0.6  # a chosen smoothing penalty (not auto-chosen)

############################################################
# 6. Start an iteration:
#    We'll do e.g. steps < 5 or so. Each iteration:
#       - compute ksiFuncs
#       - build M = Sigma + N*Lambda*I
#       - build T
#       - compute y = tilde_y(P, Fi, f0, direction=??, tSpan, F)
#         -> Actually we must do c, d -> direction -> THEN y
#       - solve for c, d
#       - update f0
############################################################

maxIter = 5
steps = 0

while steps < maxIter:
    print(f"\nIteration #{steps}")

    # 6a) Build the ksi functions
    ksi_functions = ksiFuncs(Fi, f0, F, p, tSpan)
      # shape: list of length #bonds, each is an InterpolatedUnivariateSpline or similar

    # 6b) Build M = Sigma + N*Lambda I
    M = getM(Lambda, Fi, ksi_functions, p, tSpan)  # shape (#bonds, #bonds)

    # 6c) Build T
    T = getT(p, Fi, f0, F, tSpan)                 # shape (#bonds, p)

    # 6d) We can't form tilde_y *yet* because it needs the direction = sum_j(d_j phi_j) + sum_i(c_i ksi_i).
    #     But from the paper, c and d come from:
    #
    #         aux  = (T^T M^-1 T)^-1 T^T M^-1
    #         d    = aux @ y
    #         c    = M^-1 (I - T aux) @ y
    #
    #     but y is itself the partial linearization: y_k = Pk - Nk(f0) + Dk( f0, ??? ) ???
    #
    # Actually the typical formula is:
    #    y = P - N(f0)   [the "residual"]   # plus the linear term goes into T and M steps.
    #
    # So let's define a "direction-free" residual first:
    # We'll define "resBase" = [Pk - Nk(f0)] for each bond k
    # Then "tilde_y" = resBase + [Dk( f0, direction )].
    # But in matrix form, that second part becomes T * d - M*C logic.
    # We'll do it exactly as in eq. (31)-(32).

    # Let's define a simple "y = Pk - Nk(f0)" (the zero-order step):
    # We'll pass a zero direction to tilde_y if we want the same shape, but let's do it manually:

    # Build zero direction function:
    zeroDirection = lambda x: 0.0
    # Then call tilde_y:
    baseY = tilde_y(P, Fi, f0, zeroDirection, tSpan, F)
    # "baseY" = [ Pk - Nk(f0) ] basically, since the Dk part is 0.

    # Convert to array shape (#bonds,1):
    y = baseY  # shape (#bonds,)

    # invert M:
    invM = np.linalg.pinv(M)
    # compute T^T @ invM @ T:
    mid = T.T @ invM @ T
    invMid = np.linalg.pinv(mid)

    aux = invMid @ (T.T @ invM)
    # d is shape (p,):
    d = aux @ y
    # c is shape (#bonds,):
    c = invM @ (np.eye(M.shape[0]) - (T @ aux)) @ y

    # 6e) Build the direction function for f(t) = f0(t) + direction(t).
    #     direction(t) = sum_j d[j]*phi_j(t) + sum_i c[i]*ksi_i(t).
    phiBasis = getPhiBasisFunctions(p, start=0)  # exactly p polynomials j=0..p-1

    directionVec = np.zeros_like(tSpan)
    # sum up the polynomial part:
    for j, phiFunc in enumerate(phiBasis):
        directionVec += d[j] * phiFunc(tSpan)  # no norm dividing here

    # sum up the ksi part:
    for iBond in range(N):
        directionVec += c[iBond] * ksi_functions[iBond](tSpan)

    # Make an actual function from directionVec:
    directionFunc = interp1d(tSpan, directionVec, kind='linear', fill_value="extrapolate")

    # 6f) Update f0 => new f0 = old f0 + direction
    old_f0_vals = f0(tSpan)  # shape = (#time_points,)
    new_f0_vals = old_f0_vals + directionVec
    f0 = interp1d(tSpan, new_f0_vals, kind='linear', fill_value="extrapolate")

    steps += 1

############################################################
# 7. Plot or compare final z(t)
############################################################

import matplotlib.pyplot as plt

# 1) Build final curve from the final f(t)
z_final = zVectorFromf(f0, F, tSpan)
d_final = dVectorFromf(f0, F, tSpan)  
# e.g. an array of shape (len(tSpan),)

calcYields = np.zeros(N)   # N = number of bonds
for i in range(N):
    bondPeriods = int(np.nonzero(Fi[i])[0][-1]) + 1
    # times for this bond
    localTimes = tSpan[:bondPeriods]
    # flows for bond i
    Fik = Fi[i][:bondPeriods]
    # discount subset
    discountSubset = d_final[:bondPeriods]

    # compute eq. yield
    calcYields[i] = bondYTMfromZeroDiscount(Fik, localTimes, discountSubset)

# 2) Original yields to compare
y_original = yields

# 3) Plot them
import matplotlib.pyplot as plt

finalTimes = []
for i, calendar in enumerate(cl.calendars):
    finalTimes.append( ytmCalc.dateToYears( calendar.dates[-1] ) )

plt.figure()
plt.plot(finalTimes, y_original,  'ro-', label='Original YTMcalc yields')
plt.plot(finalTimes, calcYields, 'bs-',  label='Calculated yields from Zero-Curve')
plt.xlabel("Bond Maturity (years)")
plt.ylabel("Yield to Maturity")
plt.legend()
plt.show()