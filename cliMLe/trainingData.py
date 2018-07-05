import os, itertools, datetime
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector
import matplotlib.pyplot as plt
from cliMLe.dataProcessing import PreProc
from typing import List, Union
from cliMLe.pcProject import PCDataset

class DataSource:

    def __init__(self, name, time_bounds ):
         # type: (str,(str,str)) -> None
        self.time_bounds = time_bounds
        self.name = name
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.dateRange = self.getDateRange()

    def getDataFilePath(self, type, ext ):
        # type: (str,str) -> str
        return os.path.join(os.path.join(self.rootDir, "data",type), self.name + "." + ext )

    def getDateRange(self):
        # type: () -> [ datetime.date, datetime.date ]
        return [ self.toDate( dateStr ) for dateStr in self.time_bounds ]

    def toDate( self, dateStr ):
        # type: (str) -> datetime.date
        toks = [ int(tok) for tok in dateStr.split("-") ]
        return datetime.date( toks[0], toks[1], toks[2] )

    def inDateRange( self, date ):
        # type: (datetime.date) -> bool
        return date >= self.dateRange[0] and date <= self.dateRange[1]

    def getTimeseries( self, **kwargs ): raise NotImplementedError
        # type: ( Union[bool,int] ) -> list[(str,list(datetime.date),np.ndarray)]


class ProjectDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds ):
        DataSource.__init__( self, name, time_bounds )
        self.variables = variableList
        self.selector = Selector(time=self.time_bounds )
        self.dataFile = self.getDataFilePath("cvdp","nc")
        self.variables = variableList

    def getTimeseries( self, **kwargs ):
        # type: ( Union[bool,int] ) -> list[(str,list(datetime.date),np.ndarray)]
        dset = cdms.open( self.dataFile )
        normalize = kwargs.get( "norm", True )
        smooth = kwargs.get( "smooth", 0 )
        offset = kwargs.get( "offset", 0 )
        length = kwargs.get( "length", -1 )
        norm_timeseries = []
        for varName in self.variables:
            variable =  dset.getVariable( varName )   # type: cdms.fvariable.FileVariable
            subset_variable = variable(self.selector) # type: cdms.tvariable.TransientVariable
            timeAxis = subset_variable.getTime()      # type: cdms.axis.Axis
            dates = [ datetime.date() for datetime in timeAxis.asdatetime() ]
            timeseries =  subset_variable.data  # type: np.ndarray
            if( length == -1 ): length = timeseries.shape[0] - offset
            offset_data = timeseries[offset:offset+length]
            norm_data = PreProc.normalize( offset_data ) if normalize else offset_data
            for iS in range(smooth): norm_data = PreProc.lowpass( norm_data )
            norm_timeseries.append( (varName, dates, norm_data ) )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()


class IITMDataSource(DataSource):

    def __init__(self, name, _type, time_bounds):
        DataSource.__init__(self, name, time_bounds)
        self.dataFile = self.getDataFilePath("IITM","txt")
        self.type = _type

    def getTimeseries(self, **kwargs):
        # type: ( Union[bool,int] ) -> list[(str,list(datetime.date),np.ndarray)]
        dset = open(self.dataFile,"r")
        lines = dset.readlines()
        normalize = kwargs.get("norm", True)
        smooth = kwargs.get("smooth", 0)
        offset = kwargs.get("offset", 0)
        length = kwargs.get("length", -1)
        timeseries = []
        dates = []
        for iLine in range( len(lines) ):
            line_elems = lines[iLine].split()
            if self.isYear(line_elems[0]):
                iYear = int(line_elems[0])
                for iMonth in range(1,13):
                    value = float( line_elems[ iMonth ] )
                    date = datetime.date(iYear, iMonth, 15)
                    if self.inDateRange(date):
                        timeseries.append( value )
                        dates.append( date )
        if (length == -1): length = len(timeseries) - offset
        offset_data = np.array( timeseries[offset:offset + length] )
        norm_data = PreProc.normalize(offset_data) if normalize else offset_data
        for iS in range(smooth): norm_data = PreProc.lowpass(norm_data)
        return [ ( self.type, dates, norm_data ) ]

    def isYear( self, s ):
        # type: (str) -> bool
        try:
            test_val = int(s)
            return ( test_val > 1700 and test_val < 3000)
        except ValueError:
            return False

class TrainingDataset:

    def __init__(self, sources, _inputDataset = None, **kwargs ):
        # type: ( list[DataSource], PCDataset, Union[bool, str, float] ) -> None
        self.dataSources = sources
        self.offset = _inputDataset.nTSteps - 1 if _inputDataset is not None else 0
        self.smooth = _inputDataset.smooth if _inputDataset is not None else 0
        self.data = np.column_stack([tsdata for (tsname, dates, tsdata) in self.getTimeseries()])
        self.output_size = self.data.shape[1]

    def addDataSource(self, dsource ):
        # type: (DataSource) -> None
        self.dataSources.append( dsource )

    def getTimeseries(self):
        # type: (bool) -> list[(str,np.ndarray)]
        timeseries = []
        for dsource in self.dataSources: timeseries.extend( dsource.getTimeseries( offset=self.offset, smooth=self.smooth ) )
        return timeseries

    def plotTimeseries(self):
        plt.title("Training data")
        for dsource in self.dataSources: 
            for ( tsname, dates, tsdata ) in dsource.getTimeseries( offset=self.offset, smooth=self.smooth ):
                plt.plot_date( dates, tsdata, 'b-', label = tsname )
        plt.legend()
        plt.show()

    def getEpoch(self):
        # type: () -> np.ndarray
        return self.data

if __name__ == "__main__":

#    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )
#    print "Variables: " + str( td.listVariables() )

    td = IITMDataSource( "AI", "monthly", ( "1980-1-1", "2014-12-31" )  )
    dset = TrainingDataset( [td] )
    dset.plotTimeseries()