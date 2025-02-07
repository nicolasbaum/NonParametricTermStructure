############################################################
# main.py
# 
# This version does automatic selection of lambda via GCV 
# in each iteration, rather than hardcoding it.
############################################################

import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bondcalendar import BondCalendarLoader
from phi_basis_functions import getPhiBasisFunctions
import datetime

# SciPy for brent or minimize_scalar
from scipy.optimize import minimize_scalar

# Bring in your code (make sure they're in the same folder):
from calculate_ksi import ksiFuncs
from calculate_sigma import getSigma
from calculate_T import getT
from calculate_M import getM
from tilde_y import tilde_y
from calculate_r import zVectorFromf, dVectorFromf
from YTM import YTMCalculator, bondYTMfromZeroDiscount

############################################################
# 1) The "hat matrix" A(lambda) and GCV routine
############################################################

def compute_hat_matrix(Sigma, T, lam):
    """
    Build the "hat matrix" A(lambda) = T (T^T M^-1 T)^-1 T^T M^-1
    with M=Sigma + N*lam*I, shape (N,N).
    T is shape (N, p).
    """
    N = Sigma.shape[0]
    M = Sigma + N*lam*np.eye(N)
    invM = np.linalg.inv(M)
    mid  = T.T @ invM @ T  # shape (p,p)
    invMid = np.linalg.pinv(mid)
    A = T @ (invMid @ (T.T @ invM))  # shape (N,N)
    return A

def gcv_score(Sigma, T, y, lam):
    """
    Standard GCV formula:
      GCV(lam) = || (I - A(lam)) y ||^2 / [ trace(I - A(lam)) ]^2
    where A(lam)= T(...) for the partial linearization approach.
    """
    N = Sigma.shape[0]
    A = compute_hat_matrix(Sigma, T, lam)
    IminusA = np.eye(N) - A
    r = IminusA @ y  # the "residual"
    numerator = r @ r
    traceA = np.trace(A)
    traceIminusA = N - traceA
    if abs(traceIminusA)<1e-12:
        return 1e12
    denominator = (traceIminusA)**2
    return numerator/denominator

def find_best_lambda_via_gcv(Sigma, T, y, lam_min=1e-8, lam_max=1e2):
    """
    Minimizes GCV in bracket [lam_min, lam_max].
    Return the best lam.
    """
    def obj(lam):
        return gcv_score(Sigma, T, y, lam)
    res = minimize_scalar(obj, bounds=(lam_min, lam_max), method='bounded')
    if not res.success:
        print("Warning: GCV search failed. Using fallback lam=1e-3.")
        return 1e-3
    return res.x

############################################################
# 2) Solve for c, d once we fix lambda
############################################################

def solve_for_cd(Sigma, T, lam, y):
    """
    Reproduces the system:
      M = Sigma + N lam I
      c, d from the standard eqns:
        M c + T d = y
        T^T c = 0 or the "sub-block" approach.
    Usually you do:
      c = M^-1 ( y - T d ) 
      d = (T^T M^-1 T)^-1 T^T M^-1 y
    """
    N = Sigma.shape[0]
    M = Sigma + N*lam*np.eye(N)
    invM = np.linalg.pinv(M)

    # mid = T^T invM T => shape (p,p)
    mid  = T.T @ invM @ T
    invMid = np.linalg.pinv(mid)

    d = invMid @ (T.T @ invM @ y)  # shape (p,)
    # c = invM( y - T d )
    c = invM @ ( y - T@d )  # shape (N,)

    return c, d

############################################################
# 3) Recompute the function f(t). 
#    f(t) = old_f(t) + sum_i c[i]*xi_i(t) + sum_j d[j]*phi_j(t).
#
# We'll define a helper to build that "direction" function
# then new_f0 = old_f0 + direction(t).
#
# For demonstration, we'll just do a numeric vector approach
# on tSpan. You likely have an interp1d or something.
############################################################

def update_f0(f0, c, d, ksi_functions, phiBasis, tSpan):
    """
    f0, c, d => new f0. 
    Typically:
      direction(t) = sum_i c[i]*ksi_i(t) + sum_j d[j]*phi_j(t).
    Return the updated function as an array or an interp1d.

    We'll do a shape = (len(tSpan),) array example.
    """
    old_vals = f0(tSpan)  # if f0 is a callable
    direction = np.zeros_like(old_vals)

    # add c[i]*ksi_i(tSpan)
    for iBond, cVal in enumerate(c):
        direction += cVal * ksi_functions[iBond](tSpan)

    # add d[j]*phi_j(tSpan)
    for j, dVal in enumerate(d):
        phiVals = phiBasis[j](tSpan)  # shape (len(tSpan),)
        direction += dVal*phiVals

    new_vals = old_vals + direction

    # Return as an updated function. e.g.:
    return interp1d(tSpan, new_vals, kind='cubic', fill_value='extrapolate')


############################################################
# 4) The main iterative loop
############################################################

def main():
    # 4.1) Load your F, P, tSpan, etc. 
    # E.g.:
    # from your "YTM" or else
    # we do a quick placeholder:
    bcl = BondCalendarLoader('calendars.xlsx')
    cl  = bcl.getCalendarList()
    bondPrices = bcl.getBondPrices()
    P = bondPrices['Price'].values  # shape (#bonds,)
    Fi = cl.getPaymentMatrix()      # shape (#bonds, #time_points)

    # build tSpan
    tSpan = np.arange(1, Fi.shape[1]+1)/12.0

    # 4.2) define your function F(x)= x^2, etc. 
    
    x = sympy.Symbol('x', real=True, nonnegative=True)
    F_expr = x**2
    F = sympy.lambdify(x, F_expr, "numpy")

    # build an initial guess for f0:
    # e.g. constant sqrt(0.05)
    init_vals = np.sqrt(0.05)*np.ones_like(tSpan)
    f0 = interp1d(tSpan, init_vals, kind='linear', fill_value='extrapolate')

    # fix derivative order p
    p = 2

    # 4.3) The iterative loop
    maxIter = 5
    tol = 1e-5
    old_f_vals = f0(tSpan)  
    phiBasis = getPhiBasisFunctions(p, start=0) # e.g. [phi0, phi1]

    for step in range(maxIter):
        print(f"\n=== Iteration {step} ===")
        #  (A) build ksi: 
        ksi_functions = ksiFuncs(Fi, f0, F, p, tSpan)
        #  (B) build Sigma = getSigma(Fi, p, ksi_functions, tSpan)
        Sigma = getSigma(Fi, p, ksi_functions, tSpan)
        #  (C) build T 
        T = getT(p, Fi, f0, F, tSpan)
        #  (D) partial-lin residual y= tilde_y(P, Fi, f0, direction=??? ) 
        #      but we do direction=0 => y= P - N(f0)
        #      Because we only need the zero-order term for GCV
        #      or if your code expects a direction param:
        zeroDir = lambda x: 0.0
        y = tilde_y(P, Fi, f0, zeroDir, tSpan, F)

        #  (E) auto find lambda 
        lam = find_best_lambda_via_gcv(Sigma, T, y)
        print("  -> chosen lambda =", lam)

        #  (F) now solve for c,d
        c, d = solve_for_cd(Sigma, T, lam, y)

        #  (G) update f0
        f0 = update_f0(f0, c, d, ksi_functions, phiBasis, tSpan)

        # (H) check for converge
        new_f_vals = f0(tSpan)
        diffNorm = np.linalg.norm(new_f_vals - old_f_vals)
        print("  difference in f0 =", diffNorm)
        if diffNorm<tol:
            print("Converged!")
            break
        old_f_vals = new_f_vals.copy()


    # 1) Build final curve from the final f(t)
    z_final = zVectorFromf(f0, F, tSpan)
    d_final = dVectorFromf(f0, F, tSpan)  
    # e.g. an array of shape (len(tSpan),)

    N = Fi.shape[0]  # number of bonds
    calcYields = np.zeros(N)
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
    ytmCalc = YTMCalculator(calcDate=datetime.date(2018,9,12), yearConvention=365.25)
    macaulyDurations, yields, maturities = ytmCalc.getDurationsYTMsAndMaturities(cl.calendars, bondPrices)
    y_original = yields

    # 2.1 Original maturities
    finalTimes = []
    for i, calendar in enumerate(cl.calendars):
        finalTimes.append( ytmCalc.dateToYears( calendar.dates[-1] ) )

    # 3. Plot
    plt.figure()
    plt.plot(finalTimes, y_original,  'ro-', label='Original YTMcalc yields')
    plt.plot(finalTimes, calcYields, 'bs-',  label='Calculated yields from Zero-Curve')
    plt.xlabel("Bond Maturity (years)")
    plt.ylabel("Yield to Maturity")
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()