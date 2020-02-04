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

    def getPaymentMatrix(self):
        minDate = min([min(calendar.dates) for calendar in self.calendars]+[np.datetime64(datetime.date.today(),'D')])
        maxDate = max([max(calendar.dates) for calendar in self.calendars])
        dates = pd.date_range(start=pd.to_datetime(minDate).replace(day=1), end=pd.to_datetime(maxDate).replace(day=1), freq='MS')
        pmtMatrix = np.empty((len(self.calendars),len(dates)))
        for i,calendar in enumerate(sorted(self.calendars)):
            for j,date in enumerate(dates):
                paymentsDict = { pd.to_datetime(k).replace(day=1) : v for k,v in calendar.paymentsDict.items() }
                calendarDates = [ pd.to_datetime(d).replace(day=1) for d in calendar.dates ]
                pmtMatrix[i,j] = paymentsDict[date] if date in calendarDates else 0
        return pmtMatrix

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
