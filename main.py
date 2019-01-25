import numpy as np
from sympy import Symbol, lambdify
from scipy.interpolate import UnivariateSpline
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator
from tilde_y import tilde_y
from calculate_T import getT
from calculate_M import getM
import datetime


bcl = BondCalendarLoader('calendars.xlsx')
cl = bcl.getCalendarList()
bondPrices = bcl.getBondPrices()
P = bondPrices['Price'].values
Fi = cl.getPaymentMatrix()

ytmCalc = YTMCalculator( calcDate=datetime.date(2018,9,12), yearConvention=252.0 )
yieldCurve = ytmCalc.getInterpolatedYieldCurve(cl.calendars, bondPrices)

N=Fi.shape[0]
tSpan=np.arange(1,Fi.shape[1]+1)/252.0

x = Symbol('x')
F=lambdify(x,x**2, "numpy")
invF=np.vectorize(lambdify(x,x**0.5, "numpy"))

Lambda=0.6
p=2     #derivative degree used in smoothness equation

"""Since Spot curve from yield curve requires some form of bootstrap
and this isn't easy because settlement dates are not overlapping,
I'm going to use as a (bad) proxy, the yields as the spots.
"""
r0 = yieldCurve( tSpan )+0.1/100

f0Vector=invF(np.diff(r0 * tSpan) / np.diff(tSpan))
f0Vector=np.concatenate([[f0Vector[0]], f0Vector])
f0=UnivariateSpline(np.concatenate([[0],tSpan]), np.concatenate( [[0], f0Vector ]))


steps = 0
while steps < 4:
    #Idea behind f=f0 is that I want to converge to the final result where previous iteration = current f
    M = getM(Lambda, Fi, f0, F, p, tSpan )
    T = getT(p, Fi, f0, F, tSpan)
    y= tilde_y(P, Fi, f0, tSpan,F)

    #Check original code that uses QR decomposition
    invM = np.linalg.inv(np.asmatrix(M))
    aux = np.linalg.inv( T.transpose()*invM*T )*T.transpose()*invM
    aux2 = T * aux
    d=aux*y
    c=invM *( np.eye(len(aux2)) - aux2 ) *y