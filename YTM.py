import numpy as np
import sympy as sp
from scipy import optimize, interpolate
import datetime


class YTMCalculator(object):
    def __init__(self, calcDate=None, monthConvention=None, yearConvention=None):
        self.calcDate = calcDate or datetime.date.today()
        self.daysInMonth = monthConvention or 30
        self.daysInYear = yearConvention or 360
        self.ytms = None

    def calculate(self, paymentsDict, price):
        def f(ytm):
            return price - self.npv(paymentsDict, ytm)

        return optimize.brentq(f, -1, 1)

    def npv(self, paymentsDict, ytm):
        result = 0
        for date, payment in paymentsDict.items():
            if date < self.calcDate:
                continue
            yearsToDate = self.daysToDate(self.calcDate, date) / self.daysInYear
            result += payment * np.exp(-ytm * yearsToDate)
        return result

    def macaulayDuration(self, paymentsDict, ytm, bondPrice):
        result = 0
        for date, payment in paymentsDict.items():
            if date < self.calcDate:
                continue
            yearsToDate = self.daysToDate(self.calcDate, date) / self.daysInYear
            result += payment * yearsToDate * np.exp(-ytm * yearsToDate)
        return result / bondPrice

    @classmethod
    def daysToDate(cls, startDate, endDate):
        return np.busday_count(startDate, endDate)

    def dateToYears(self, date):
        days = self.daysToDate(self.calcDate, date)
        return days / self.daysInYear

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

        return tuple([np.array(x) for x in (
            durationsInYears, yields, maturitiesInYears)])  # Cast to numpy arrays so no need to cast in the future

    def getInterpolatedYieldCurve(self, bondCalendars, bondPrices):
        """Returned variable is an interpolated object"""
        _, yields, maturitiesInYears = self.getDurationsYTMsAndMaturities(bondCalendars, bondPrices)
        maturitiesInYears = [0] + maturitiesInYears
        yields = [0] + yields

        return interpolate.interp1d(maturitiesInYears, yields, bounds_error=False, fill_value='extrapolate',
                                    kind='linear')


class NelsonSiegelModel:
    def __init__(self, maturities, yields):
        self.beta0 = None
        self.beta1 = None
        self.beta2 = None
        self.tau = None
        self.fit(maturities, yields)

    @staticmethod
    def nelson_siegel(t, beta0, beta1, beta2, tau):
        """
        Nelson-Siegel model for the term structure of interest rates.
        Modified version to avoid breaking the curve at t=0.
        """
        term1 = beta0
        term2 = beta1 * (1 - np.exp(-t / tau)) / (1 - np.exp(-t / tau))
        term3 = beta2 * ((1 - np.exp(-t / tau)) / (1 - np.exp(-t / tau)) - np.exp(-t / tau))
        return term1 + term2 + term3

    def fit(self, maturities, yields):
        initial_guess = [0.03, 0.01, 0.01, 1.0]
        params, _ = optimize.curve_fit(self.nelson_siegel, maturities, yields, p0=initial_guess)
        self.beta0, self.beta1, self.beta2, self.tau = params

    def sympy_implementation(self):
        if self.beta0 is None or self.beta1 is None or self.beta2 is None or self.tau is None:
            raise ValueError("Parameters not set. Fit the model before creating the Sympy implementation.")

        t = sp.Symbol('t')
        term1 = self.beta0
        term2 = self.beta1 * (1 - sp.exp(-t / self.tau)) / (1 - sp.exp(-t / self.tau))
        term3 = self.beta2 * ((1 - sp.exp(-t / self.tau)) / (1 - sp.exp(-t / self.tau)) - sp.exp(-t / self.tau))
        return term1 + term2 + term3
