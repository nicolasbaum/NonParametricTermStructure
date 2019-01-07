import numpy as np
from sympy import Symbol, lambdify
from scipy.interpolate import UnivariateSpline
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator
from H_operator import Hi
from tilde_y import tilde_yk
from calculate_sigma import getSigma
from calculate_T import getT
from calculate_eta import eta_k
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
r0 = yieldCurve( tSpan )

f0Vector=invF(np.diff(r0 * tSpan) / np.diff(tSpan))
f0Vector=np.concatenate([[f0Vector[0]], f0Vector])
f0=UnivariateSpline(np.concatenate([[0],tSpan]), np.concatenate( [[0], f0Vector ]))

#######Just for check#######
#Reconverting f0 into yieldCurve just to check everything is working

steps = 0
while steps < 4:
    #H = [ Hi(f0, F, tSpan[:i]) for i in np.arange(len(tSpan))+1 ]
    #Idea behind f=f0 is that I want to converge to the final result where previous iteration = current f
    sigma = getSigma( Fi, f0, F, p, tSpan)
    # T = getT(p, Fi, f0, F, tSpan)
    # for calendar in cl.calendars:
    #     Pk = bondPrices.loc(bon)
    # yk = tilde_yk(Pk, Fik, f0, f0, tSpan, F)  # This should be vectorized bond per bond with Fik and Pk
    # etak = eta_k(t,Fik,f0,F,p,tRange)   #Iterate bonds and another degree of vectorization with time t