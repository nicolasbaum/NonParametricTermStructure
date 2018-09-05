import numpy as np
import pandas as pd
from collections import defaultdict
import datetime

BOND_TICKERS = [
                'AO20',
                'AA21',
                'AY24',
                'A2E2',
                'AA25',
                'AA26',
                'A2E7',
                'DICA',
                'AA37',
                'PARA',
                'AA46',  
            ]

class Calendar(object):
    def __init__(self, df):
        self.dates = self._tranformDates( df['Date'].values )
        self.payments = df['Total'].values
        self.paymentsDict = {d:p for d,p in zip(self.dates, self.payments)}

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
        dates = pd.bdate_range(minDate, maxDate)
        dates = [d.to_datetime64().astype('datetime64[D]') for d in dates]
        pmtMatrix = np.empty((len(self.calendars),len(dates)))
        for i,calendar in enumerate(self.calendars):
            for j,date in enumerate(dates):
                pmtMatrix[i,j]=calendar.paymentsDict[date] if date in calendar.dates else 0
        return pmtMatrix

class BondCalendarLoader(object):
    def __init__(self, xlsPath):
        self.xlsPath = xlsPath
        self.calendarDFs = pd.read_excel(self.xlsPath, sheet_name=None)
    
    def getCalendarList(self):
        self.calendarDict = { k:self.calendarDFs[k] for k in BOND_TICKERS }
        return ListOfCalendars( [ Calendar(self.calendarDFs[bondName]) for bondName in BOND_TICKERS ] )

    def getBondPrices(self):
        pricesDF = self.calendarDFs.get('Prices').loc[:,['Bond','Price']]
        pricesDF.set_index('Bond')
        pricesDF.reindex(BOND_TICKERS)
        return pricesDF['Price'].values
