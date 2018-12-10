import numpy as np
from sympy import Symbol, lambdify
from scipy import interpolate
from bondcalendar import BondCalendarLoader
from YTM import YTMCalculator
import datetime


bcl = BondCalendarLoader('calendars.xlsx')
cl = bcl.getCalendarList()
bondPrices = bcl.getBondPrices()
P = bondPrices['Price'].values
Fi = cl.getPaymentMatrix()

ytmCalc = YTMCalculator( calcDate=datetime.date(2018,9,12) )
yieldCurve = { calendar.dates[-1] : ytmCalc.calculate( calendar.paymentsDict,
                                                bondPrices[bondPrices['Bond']==calendar.bondName].Price.values[0]
                                                )
         for calendar in cl.calendars }

#ToDo: Calculate zero-coupon curve with yield curve

N=Fi.shape[0]
tSpan=np.arange(1,Fi.shape[1]+1)/252.0

x = Symbol('x')
F=np.vectorize(lambdify(x,x**2, "numpy"))
invF=np.vectorize(lambdify(x,x**0.5, "numpy"))

Lambda=0.6
r0=0.1

#Verify this crap
f0vector=invF(r0*np.diff(tSpan))
f0vector=np.concatenate([[f0vector[0]],f0vector])
f0=interpolate.interp1d(tSpan,f0vector)
print(f0vector)











