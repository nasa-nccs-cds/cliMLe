import os, sys, math, datetime
from cdms2.selectors import Selector
from cliMLe.svd import Eof
import numpy as np
import pandas as pd
import xarray as xr

class Analytics:
    smoothing_kernel = np.array([.13, .23, .28, .23, .13])
    months = "jfmamjjasonjfmamjjason"

    @staticmethod
    def normalize( data, axis=0 ):
        # type: (np.ndarray) -> np.ndarray
        std = np.std( data, axis )
        return data / std

    @staticmethod
    def center( data, axis=0 ):
        # type: (np.ndarray) -> np.ndarray
        mean = np.average( data, axis )
        return data - mean

    @staticmethod
    def getMonthFilterIndices(filter):
        try:
            start_index = int(filter)
            return (start_index, 1)
        except:
            start_index = "xjfmamjjasondjfmamjjasond".index(filter.lower())
            if start_index < 0: raise Exception("Unrecognizable filter value: " + filter )
            return ( start_index, len(filter) )

    @classmethod
    def lowpass( cls, data ):
        # type: (np.ndarray) -> np.ndarray
        return np.convolve( data, cls.smoothing_kernel, "same")

    @classmethod
    def decycle( cls, dates, data ):
        # type: (list[datetime.date],np.ndarray) -> np.ndarray
        if len(data.shape) == 1: data = data.reshape( [ data.shape[0], 1 ] )
        times = pd.DatetimeIndex( data=dates, name="time" )                                             # type: pd.DatetimeIndex
        ds = xr.Dataset({'data': ( ('time', 'series'), data) },   {'time': times } )                    # type: xr.Dataset
        climatology = ds.groupby('time.month').mean('time')                                             # type: xr.Dataset
        anomalies = ds.groupby('time.month') - climatology                                              # type: xr.Dataset
        return anomalies["data"].data

    @classmethod
    def orthoModes( cls, data, nModes ):
        # type: (np.ndarray, int) -> np.ndarray
        eof = Eof( data, None, False, False )
        result = eof.eofs( 0, nModes )  # type: np.ndarray
        return result.transpose()

    @classmethod
    def yearlyAve( cls, start_date, freq, data ):
        # type: (CDate,str,np.ndarray) -> np.ndarray
        if len(data.shape) == 1: data = data.reshape( [ data.shape[0], 1 ] )
        times = pd.date_range(start=str(start_date), periods=data.shape[0], freq=freq, name='time')     # type: pd.DatetimeIndex
        ds = xr.Dataset({'data': ( ('time', 'series'), data) },   {'time': times } )                    # type: xr.Dataset
        yearly_ave = ds.groupby('time.year').mean('time')                                               # type: xr.Dataset
        return yearly_ave["data"].data


    @classmethod
    def monthSubsetIndices( cls, subset ):
        start_index = cls.months.index( subset.lower() ) + 1
        return range( start_index, start_index + len(subset) )

    # @classmethod
    # def monthlySubset( cls, start_date, subset, data ):
    #     # type: (CDate,str,np.ndarray) -> np.ndarray
    #     subset_indices = cls.monthSubsetIndices( subset )
    #     if len(data.shape) == 1: data = data.reshape( [ data.shape[0], 1 ] )
    #     times = pd.date_range(start=str(start_date), periods=data.shape[0], freq=freq, name='time')     # type: pd.DatetimeIndex
    #     ds = xr.Dataset({'data': ( ('time', 'series'), data) },   {'time': times } )                    # type: xr.Dataset
    #     yearly_ave = ds.groupby('time.year').mean('time')                                               # type: xr.Dataset
    #     return yearly_ave["data"].data

class CDuration:

    MONTH = "M"
    YEAR = "Y"

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

    def extend(self, duration ):
        # type: (CDuration) -> CTimeRange
        return CTimeRange(self.startDate, self.endDate.inc(duration) )

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

if __name__ == "__main__":
    print str( Analytics.monthSubsetIndices( "JFMA") )