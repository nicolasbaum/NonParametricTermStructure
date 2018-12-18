import numpy as np
from sympy import Symbol, lambdify
from scipy import interpolate
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator
from H_operator import Hi
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
F=np.vectorize(lambdify(x,x**2, "numpy"))
invF=np.vectorize(lambdify(x,x**0.5, "numpy"))

Lambda=0.6

"""Since Spot curve from yield curve requires some form of bootstrap
and this isn't easy because settlement dates are not overlapping,
I'm going to use as a (bad) proxy, the yields as the spots.
"""
r0 = yieldCurve( tSpan )

f0Vector=invF(np.diff(r0 * tSpan) / np.diff(tSpan))
f0Vector=np.concatenate([[f0Vector[0]], f0Vector])
f0=interpolate.interp1d(tSpan, f0Vector)
#Reconverting f0 into yieldCurve just to check everything is working
rVector=np.concatenate([[0],np.array([ Hi(f0,F,tSpan[:i]) for i in range(1,len(tSpan)) ])])/tSpan
