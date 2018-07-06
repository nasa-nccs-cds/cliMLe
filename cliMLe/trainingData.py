import os, itertools, datetime
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector
import matplotlib.pyplot as plt
from cliMLe.dataProcessing import PreProc, CTimeRange, CDuration
from typing import List, Union, Dict, Any
from cliMLe.pcProject import PCDataset

class TimeseriesData:

    def __init__(self, _dates, _series ):
        # type: (list(datetime.date), list[(str,np.ndarray)]) -> None
        self.dates = _dates
        self.series = { k:v for (k,v) in _series }    # type: Dict[str,np.ndarray]
        self.data = np.column_stack( self.series.values() )
        self.output_size = len( self.series.keys() )

class DataSource:

    def __init__(self, name ):
         # type: (str) -> None
        self.name = name
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def getDataFilePath(self, type, ext ):
        # type: (str,str) -> str
        return os.path.join(os.path.join(self.rootDir, "data",type), self.name + "." + ext )

    def getTimeseries( self, timeRange, **kwargs ): raise NotImplementedError
        # type: ( Union[bool,int] ) -> TimeseriesData


class ProjectDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds ):
        DataSource.__init__( self, name )
        self.variables = variableList
        self.dataFile = self.getDataFilePath("cvdp","nc")
        self.variables = variableList

    def getTimeseries( self, timeRange, **kwargs ):
        # type: ( Union[bool,int] ) -> TimeseriesData
        dset = cdms.open( self.dataFile )
        normalize = kwargs.get( "norm", True )
        smooth = kwargs.get( "smooth", 0 )
        offset = kwargs.get( "offset", 0 )
        length = kwargs.get( "length", -1 )
        norm_timeseries = []
        dates = None
        selector = timeRange.getSelector() if timeRange else None
        for varName in self.variables:
            variable =  dset( varName, selector ) if selector else dset( varName ) # type: cdms.tvariable.TransientVariable
            if dates is None:
                timeAxis = variable.getTime()      # type: cdms.axis.Axis
                dates = [ datetime.date() for datetime in timeAxis.asdatetime() ]
            timeseries =  variable.data  # type: np.ndarray
            if( length == -1 ): length = timeseries.shape[0] - offset
            offset_data = timeseries[offset:offset+length]
            norm_data = PreProc.normalize( offset_data ) if normalize else offset_data
            for iS in range(smooth): norm_data = PreProc.lowpass( norm_data )
            norm_timeseries.append( (varName, norm_data ) )
        dset.close()
        return TimeseriesData( dates, norm_timeseries )

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()


class IITMDataSource(DataSource):

    def __init__(self, name, _type):
        # type: ( str, str ) -> None
        DataSource.__init__(self, name)
        self.type = _type.lower()
        self.dataFile = self.getDataFilePath("IITM","txt")

    def getTypeIndices(self):
        if self.type.startswith("ann"):
            return 13, 6
        elif self.type == "jf":
            return 14, 2
        elif self.type == "mam":
            return 15, 4
        elif self.type == "jjas" or self.type == "monsoon":
            return 16, 8
        elif self.type == "ond":
            return 17, 11
        else:
            raise Exception( "Unrecognized Data source type: " + self.type )

    def getTimeseries(self, timeRange, **kwargs):
        # type: ( Union[bool,int] ) -> TimeseriesData
        dset = open(self.dataFile,"r")
        lines = dset.readlines()
        normalize = kwargs.get("norm", True)
        smooth = kwargs.get("smooth", 0)
        timeseries = []
        dates = []
        for iLine in range( len(lines) ):
            line_elems = lines[iLine].split()
            if self.isYear(line_elems[0]):
                iYear = int(line_elems[0])
                if self.type == "monthly":
                    for iMonth in range(1,13):
                        value = float( line_elems[ iMonth ] )
                        date = datetime.date(iYear, iMonth, 15)
                        if  not timeRange or timeRange.inDateRange(date):
                            timeseries.append( value )
                            dates.append( date )
                else:
                    colIndex, iMonth = self.getTypeIndices()
                    value = float( line_elems[ colIndex ] )
                    date = datetime.date(iYear, iMonth, 15)
                    if not timeRange or timeRange.inDateRange(date):
                        timeseries.append( value )
                        dates.append( date )
        tsarray = np.array( timeseries )
        norm_data = PreProc.normalize(tsarray) if normalize else tsarray
        for iS in range(smooth): norm_data = PreProc.lowpass(norm_data)
        return TimeseriesData( dates, [ ( self.type, norm_data ) ] )

    def isYear( self, s ):
        # type: (str) -> bool
        try:
            test_val = int(s)
            return ( test_val > 1700 and test_val < 3000)
        except ValueError:
            return False

class TrainingDataset:

    def __init__(self, sources, _predictionLag, **kwargs ):
        # type: ( list[DataSource], CDuration ) -> None
        self.dataSources = sources
        self.smooth = kwargs.get("smooth",0)
        self.trainingRange = kwargs.get( "trainingRange", None ) # type: CTimeRange
        self._timeseries = []

    @classmethod
    def new(cls, sources, inputDataset, predictionLag, **kwargs):
        # type: ( list[DataSource], PCDataset, CDuration ) -> TrainingDataset
        offset = inputDataset.nTSteps - 1 if inputDataset is not None else 0
        smooth = inputDataset.smooth if inputDataset is not None else 0
        trainingRange = inputDataset.timeRange.shift(predictionLag.inc(offset))
        return cls( sources, predictionLag, smooth=smooth, trainingRange=trainingRange )

    def addDataSource(self, dsource ):
        # type: (DataSource) -> None
        self.dataSources.append( dsource )

    def getTimeseries(self):
        # type: () -> ( list[TimeseriesData] )
        if len( self._timeseries ) == 0:
            self._timeseries = [ dsource.getTimeseries( self.trainingRange, smooth=self.smooth ) for dsource in self.dataSources]  # type: List[TimeseriesData]
        return self._timeseries

    def getTimeseriesData(self):
        # type: (bool) -> ( list(datetime.date), list[np.ndarray] )
        timeseries = self.getTimeseries() # type: List[TimeseriesData]
        tsdata = []
        for ts in timeseries: tsdata.extend( ts.series.values())
        return ( timeseries[0].dates, tsdata )

    def plotTimeseries( self ):
        plt.title("Training data")
        timeseries = self.getTimeseries() # type: List[TimeseriesData]
        for tsdata in timeseries:
            for tsname, series in tsdata.series.items():
                plt.plot_date( tsdata.dates, series, 'b-', label = tsname )
        plt.legend()
        plt.show()

    def getEpoch( self ):
        # type: () -> ( list[dates], np.ndarray )
        dates, series = self.getTimeseriesData()
        return dates, np.column_stack( series )

    def getOutputSize(self):
        timeseries = self.getTimeseries() # type: List[TimeseriesData]
        return sum( [ ts.output_size for ts in timeseries ] )

if __name__ == "__main__":

#    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )
#    print "Variables: " + str( td.listVariables() )

    td = IITMDataSource( "AI", "monsoon"  )
    dset = TrainingDataset( [td], 0 )
    dset.plotTimeseries()