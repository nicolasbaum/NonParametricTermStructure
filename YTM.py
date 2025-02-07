import numpy as np
from scipy import optimize, interpolate
import datetime

class YTMCalculator(object):
    def __init__(self, calcDate=None,monthConvention=None, yearConvention=None):
        self.calcDate = calcDate or datetime.date.today()
        self.daysInMonth = monthConvention or 30
        self.daysInYear = yearConvention or 360
        self.ytms=None

    def calculate( self, paymentsDict, price):
        def f(ytm):
            return price-self.npv(paymentsDict, ytm)
        return optimize.brentq(f, -0.999999, 20.0)

    def npv(self, paymentsDict, ytm):
        result = 0
        for date, payment in paymentsDict.items():
            if date<self.calcDate:
                continue
            yearsToDate = self.daysToDate( self.calcDate, date )/self.daysInYear
            result += payment*np.exp(-ytm*yearsToDate)
        return result

    def macaulayDuration(self, paymentsDict, ytm, bondPrice):
        result = 0
        for date, payment in paymentsDict.items():
            if date<self.calcDate:
                continue
            yearsToDate = self.daysToDate( self.calcDate, date )/self.daysInYear
            result += payment*yearsToDate*np.exp(-ytm*yearsToDate)
        return result/bondPrice

    @classmethod
    def daysToDate(cls, startDate, endDate):
        return np.busday_count(startDate, endDate)

    def dateToYears(self, date):
        days = self.daysToDate(self.calcDate,date)
        return days/self.daysInYear

    def getDurationsYTMsAndMaturities(self, bondCalendars, bondPrices):
        maturitiesInYears = []
        durationsInYears = []
        yields = []

        for i, calendar in enumerate(bondCalendars):
            maturitiesInYears.append(self.dateToYears(calendar.dates[-1]))
            bondPrice = bondPrices.loc[calendar.bondName].Price
            ytm = self.calculate(calendar.paymentsDict, bondPrice)
            durationsInYears.append(self.macaulayDuration(calendar.paymentsDict, ytm, bondPrice))
            yields.append(ytm)

        return tuple([np.array(x) for x in (durationsInYears, yields, maturitiesInYears)]) #Cast to numpy arrays so no need to cast in the future

    def getInterpolatedYieldCurve(self, bondCalendars, bondPrices):
        """Returned variable is an interpolated object"""
        _, yields, maturitiesInYears = self.getDurationsYTMsAndMaturities(bondCalendars, bondPrices)
        maturitiesInYears = [0] + maturitiesInYears
        yields = [0] + yields

        return interpolate.interp1d( maturitiesInYears, yields, bounds_error=False, fill_value='extrapolate',kind='linear' )
    
def bondYTMfromZeroDiscount(Fik, times, discountFactors, bracket=(-1.0, 500.0)):
    """
    Given:
      Fik: shape (m,) row of bond i's cashflows at times[0..m-1]
      times: shape (m,) the time in years for columns
      discountFactors: shape(m,) the discount factors from final zero curve => d_j = d(t_j)
    
    We compute:
       P_calc = sum_j Fik[j] * discountFactors[j]
    Then solve for y_i s.t.
       sum_j [ Fik[j] * exp(-y_i * times[j]) ] == P_calc
    
    returns y_i (the single yield that reproduces the same price).
    """
    # 1) The theoretical price from the zero curve
    P_calc = np.sum(Fik * discountFactors)

    # 2) function for brent root
    def f(y):
        # single yield discount for all flows
        return np.sum(Fik * np.exp(-y * times)) - P_calc

    # 3) solve f(y)=0
    # note bracket can be bigger if needed
    # if the bond is overpriced or underpriced, might see sign issues
    ySolution = optimize.brentq(f, bracket[0], bracket[1])
    return ySolution