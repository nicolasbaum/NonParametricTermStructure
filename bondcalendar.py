import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import functools

BOND_TICKERS = [
                'AO20',
                'AA21',
                'A2E2',
                'AY24',
                'AA25',
                'AA26',
                'A2E7',
                'DICA',
                'AA37',
                'PARA',
                'AA46',  
            ]

@functools.total_ordering
class Calendar(object):
    def __init__(self, df, bondName):
        self.bondName = bondName
        self.dates = self._tranformDates( df['Date'].values )
        self.payments = df['Total'].values
        self.paymentsDict = {d:p for d,p in zip(self.dates, self.payments)}

    @property
    def _maturity(self):
        return max(self.dates)

    def __lt__(self, other):
        return self._maturity < other._maturity

    def __eq__(self, other):
        return self._maturity == other._maturity

    def _tranformDates(self, Dates):
        result = []
        for date in Dates:
            if isinstance( date, str ):
                day,month,year = [ int(component) for component in date.split('/') ]
                if year<100:
                    year+=2000
                result.append( np.datetime64(datetime.date(year=year, month=month, day=day), 'D') )
            else:
                result.append( date.astype('datetime64[D]') )
        return result

class ListOfCalendars(object):
    def __init__(self, calendars):
        self.calendars = calendars
        self.payments = defaultdict(tuple)
        self._populatePayments()

    def _populatePayments(self):
        for calendar in self.calendars:
            for date in calendar.dates:
                self.payments[date]+=calendar.paymentsDict[date]

    def getPaymentMatrixAndWarpedTimeSpanInYears(self,maxSamples=300):
        minDate = min([min(calendar.dates) for calendar in self.calendars]+[np.datetime64(datetime.date.today(),'D')])
        maxDate = max([max(calendar.dates) for calendar in self.calendars])
        daysDifference=(maxDate-minDate)/np.timedelta64(1,'D')
        warpedTspanInDays=np.geomspace(start=1,stop=daysDifference+10,num=maxSamples)

        pmtMatrix = np.zeros((len(self.calendars),len(warpedTspanInDays)))
        for i,calendar in enumerate(sorted(self.calendars)):
            for date,payment in calendar.paymentsDict.items():
                numberOfDays=(pd.to_datetime(date) - minDate) / np.timedelta64(1, 'D')
                j=np.searchsorted(warpedTspanInDays,numberOfDays)
                pmtMatrix[i,j] += payment
        return (pmtMatrix,warpedTspanInDays/365.25)

class BondCalendarLoader(object):
    def __init__(self, xlsPath):
        self.xlsPath = xlsPath
        self.calendarDFs = pd.read_excel(self.xlsPath, sheet_name=None)
        self.cl = None

    def getCalendarList(self):
        if not self.cl:
            self.calendarDict = { k:self.calendarDFs[k] for k in BOND_TICKERS }
            self.cl = ListOfCalendars( [ Calendar(self.calendarDFs[bondName], bondName) for bondName in BOND_TICKERS ] )
        return self.cl

    def getBondPrices(self):
        pricesDF = self.calendarDFs.get('Prices').loc[:,['Bond','Price']]
        pricesDF = pricesDF.set_index('Bond')
        pricesDF = pricesDF.reindex(BOND_TICKERS)
        tickersOrderedByMaturity = [ cal.bondName for cal in sorted(self.cl.calendars) ]
        return pricesDF.loc[tickersOrderedByMaturity]