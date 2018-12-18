import numpy as np
from scipy import optimize, interpolate
import datetime

class YTMCalculator(object):
    def __init__(self, calcDate=None,monthConvention=None, yearConvention=None):
        self.calcDate = calcDate or datetime.date.today()
        self.daysInMonth = monthConvention or 30
        self.daysInYear = yearConvention or 360

    def calculate( self, paymentsDict, price):
        def f(ytm):
            return price-self.npv(paymentsDict, ytm)
        return optimize.brentq(f,-1,1)

    def npv(self, paymentsDict, ytm):
        result = 0
        for date, payment in paymentsDict.items():
            if date<self.calcDate:
                continue
            days = self.daysToDate( self.calcDate, date )
            result += payment*np.exp(-ytm*days/self.daysInYear)
        return result

    @classmethod
    def daysToDate(cls, startDate, endDate):
        return np.busday_count(startDate, endDate)

    def dateToYears(self, date):
        days = self.daysToDate(self.calcDate,date)
        return days/self.daysInYear

    def getInterpolatedYieldCurve(self, bondCalendars, bondPrices):
        """Returned variable is an interpolated object"""
        maturitiesInYears = [0]
        yields = [0]
        for i,calendar in enumerate(bondCalendars):
            maturitiesInYears.append(self.dateToYears( calendar.dates[-1]) )
            yields.append( self.calculate(calendar.paymentsDict,
                                          bondPrices[bondPrices['Bond'] == calendar.bondName].Price.values[0]) )

        return interpolate.interp1d( maturitiesInYears, yields, bounds_error=False, fill_value='extrapolate',kind='cubic' )