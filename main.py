import numpy as np
from sympy import Symbol, lambdify
from scipy.interpolate import UnivariateSpline
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator
from tilde_y import tilde_y
from calculate_T import getT
from calculate_M import getM
from calculate_ksi import ksiFuncs
from calculate_r import zVectorFromf
from phi_basis_functions import getPhiBasisFunctions
from scalar_product import getNormOfFunction
from copy import deepcopy
import datetime


bcl = BondCalendarLoader('calendars.xlsx')
cl = bcl.getCalendarList()
bondPrices = bcl.getBondPrices()
P = bondPrices['Price'].values
Fi,warpedTSpanInYears = cl.getPaymentMatrixAndWarpedTimeSpanInYears()

ytmCalc = YTMCalculator( calcDate=datetime.date(2018,9,12), yearConvention=252.0 )
yieldCurve = ytmCalc.getInterpolatedYieldCurve(cl.calendars, bondPrices)

macaulyDurations,_,_ = ytmCalc.getDurationsYTMsAndMaturities(cl.calendars, bondPrices)
scaleFactors = 1/macaulyDurations
for i,sf in enumerate(scaleFactors):
    P[i]*=sf
    Fi[i,:]*=sf

N=Fi.shape[0]
#tSpan=np.arange(1,Fi.shape[1]+1)/12.0

x = Symbol('x')
F=lambdify(x,x**2, "numpy")
invF=np.vectorize(lambdify(x,x**0.5, "numpy"))

Lambda=0.6
p=2     #derivative degree used in smoothness equation

"""Since Spot curve from yield curve requires some form of bootstrap
and this isn't easy because settlement dates are not overlapping,
I'm going to use as a (bad) proxy, the yields as the spots.
"""
r0 = yieldCurve( warpedTSpanInYears )
f0=UnivariateSpline(np.concatenate([[0],tSpan]), np.concatenate( [[0], np.nan_to_num(invF(r0)) ]))
f0Basis = getPhiBasisFunctions(p+1,1)

z0 = zVectorFromf(f0,F,tSpan)

steps = 0
while steps < 3:
    print('Iteration #{}'.format(steps))
    #Idea behind f=f0 is that I want to converge to the final result where previous iteration = current f
    ksi_functions = ksiFuncs( Fi, f0, F, p, tSpan )
    M = getM(Lambda,Fi,ksi_functions,p,tSpan)
    T = getT(p, Fi, f0, F, tSpan)
    y= tilde_y(P, Fi, f0, tSpan,F)

    #Check original idea that uses QR decomposition
    invM = np.linalg.inv(np.asmatrix(M))
    aux = np.linalg.inv( T.transpose()@invM@T )@T.transpose()@invM
    aux2 = T @ aux
    d=np.squeeze( np.array( aux@y ) )
    c=np.squeeze( np.array( invM @( np.eye(len(aux2)) - aux2 ) @y ) )

    previous_f = deepcopy( f0 )
    phiSum = np.sum( [ d[basisIndex]*f0BasisFunc(tSpan)/getNormOfFunction(f0BasisFunc, tSpan, p) for basisIndex,f0BasisFunc in enumerate(f0Basis) ], axis=0 )
    ksiSum = np.sum( [ c[ksiIndex]*ksiFunc(tSpan) for ksiIndex,ksiFunc in enumerate(ksi_functions) ], axis=0)
    f0Vector = phiSum+ksiSum
    f0 = UnivariateSpline(np.concatenate([[0],tSpan]), np.concatenate( [[0], f0Vector ]))
    steps+=1

z0Calculated = zVectorFromf(f0,F,tSpan)
from matplotlib import pyplot as plt
plt.plot(tSpan, z0,'b')
plt.plot(tSpan, z0Calculated, 'g')
plt.show()



# from matplotlib import pyplot as plt
# from matplotlib.pyplot import cm
# color=iter(cm.rainbow(np.linspace(0,1,len(f0Basis))))
#
# for phiFunc in f0Basis:
#     phi = phiFunc(tSpan)/getNormOfFunction(phiFunc, tSpan, p)
#     plt.plot(tSpan, phi, c=next(color))