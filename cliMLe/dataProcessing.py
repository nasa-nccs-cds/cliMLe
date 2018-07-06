import os, sys, math, datetime
from cdms2.selectors import Selector
import numpy as np

class PreProc:
    smoothing_kernel = np.array([.13, .23, .28, .23, .13])

    @staticmethod
    def normalize( data, axis=0 ):
        # type: (np.ndarray) -> np.ndarray
        std = np.std( data, axis )
        return data / std

    @classmethod
    def lowpass( cls, data ):
        # type: (np.ndarray) -> np.ndarray
        return np.convolve( data, cls.smoothing_kernel, "same")

class CDuration:

    MONTH = 0
    YEAR = 1

    def __init__(self, _length, _unit):
        self.length = _length
        self.unit = _unit

    @classmethod
    def months(cls, length):
        return CDuration( length, cls.MONTH )

    @classmethod
    def years(cls, length):
        return CDuration( length, cls.YEAR )

    def inc( self, increment ):
        # type: (int) -> CDuration
        return CDuration(self.length + increment, self.unit)

    def __add__(self, other):
        # type: (CDuration) -> CDuration
        assert self.unit == other.unit, "Incommensurable units in CDuration add operation"
        return CDuration( self.length + other.length , self.unit )


    def __sub__(self, other):
        # type: (CDuration) -> CDuration
        assert self.unit == other.unit, "Incommensurable units in CDuration sub operation"
        return CDuration(self.length - other.length, self.unit )

class CDate:

    def __init__(self, Year, Month, Day):
        # type: (int,int,int) -> None
        self.year  = Year
        self.month = Month
        self.day   = Day

    @classmethod
    def new(cls, date_str):
        # type: (str) -> CDate
        '''Call as: d = Date.from_str('2013-12-30')
        '''
        year, month, day = map(int, date_str.split('-'))
        return cls(year, month, day)

    def __str__(self):
        # type: () -> str
        return "-".join( map(str, [self.year,self.month,self.day] ) )

    def inc(self, duration ):
        # type: (CDuration) -> CDate
        if duration.unit == CDuration.YEAR:
            return CDate( self.year + duration.length, self.month, self.day )
        elif duration.unit == CDuration.MONTH:
            month_inc = ( self.month + duration.length - 1 )
            new_month = ( month_inc % 12 ) + 1
            new_year = self.year + month_inc / 12
            return CDate( new_year, new_month, self.day )
        else: raise Exception( "Illegal unit value: " + str(duration.unit) )

class CTimeRange:

    def __init__(self, start, end ):
        # type: (CDate,CDate) -> None
        self.startDate = start
        self.endDate = end
        self.dateRange = self.getDateRange()

    @classmethod
    def new(cls, start, end ):
        # type: (str,str) -> CTimeRange
        return CTimeRange( CDate.new(start), CDate.new(end) )

    def shift(self, duration ):
        # type: (CDuration) -> CTimeRange
        return CTimeRange(self.startDate.inc(duration), self.endDate.inc(duration) )

    def selector(self):
        # type: () -> Selector
        return Selector( time=(str(self.startDate), str(self.endDate)) )

    def getDateRange(self):
        # type: () -> [ datetime.date, datetime.date ]
        return [ self.toDate( dateStr ) for dateStr in [ str(self.startDate), str(self.endDate) ] ]

    def toDate( self, dateStr ):
        # type: (str) -> datetime.date
        toks = [ int(tok) for tok in dateStr.split("-") ]
        return datetime.date( toks[0], toks[1], toks[2] )

    def inDateRange( self, date ):
        # type: (datetime.date) -> bool
        return date >= self.dateRange[0] and date <= self.dateRange[1]