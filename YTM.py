import numpy as np
from scipy import optimize
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