import os, itertools, datetime
import cdms2 as cdms
import numpy as np
import logging, abc, traceback
from cdms2.selectors import Selector
import matplotlib.pyplot as plt
from cliMLe.dataProcessing import Analytics, CTimeRange, CDuration
from typing import List, Union, Dict, Any
from cliMLe.pcProject import PCDataset
from cliMLe.dataProcessing import Parser

class TimeseriesData:

    def __init__(self, _dates, _series ):
        # type: (list(datetime.date), list[(str,np.ndarray)]) -> None
        self.dates = _dates
        self.series = { k:v for (k,v) in _series }    # type: Dict[str,np.ndarray]
        self.data = np.column_stack( self.series.values() )
        self.output_size = len( self.series.keys() )

class DataSource:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name ):
         # type: (str) -> None
        self.name = name
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def getDataFilePath(self, type, ext ):
        # type: (str,str) -> str
        return os.path.join(os.path.join(self.rootDir, "data",type), self.name + "." + ext )

    def getTimeseries( self, timeRange, **kwargs ): raise NotImplementedError
        # type: ( Union[bool,int] ) -> TimeseriesData

    @abc.abstractmethod
    def serialize(self,lines):
        return None

class ProjectDataSource(DataSource):

    def __init__(self, name, variableList ):
        DataSource.__init__( self, name )
        self.variables = variableList
        self.dataFile = self.getDataFilePath("cvdp","nc")
        self.variables = variableList

    def getTimeseries( self, timeRange, **kwargs ):
        # type: ( CTimeRange ) -> TimeseriesData
        dset = cdms.open( self.dataFile )
        normalize = kwargs.get( "norm", True )
        smooth = kwargs.get( "smooth", 0 )
        offset = kwargs.get( "offset", 0 )
        length = kwargs.get( "length", -1 )
        norm_timeseries = []
        dates = None
        selector = timeRange.selector() if timeRange else None
        logging.info( "TrainingData: variables = {0}, time range = {1}".format( str(self.variables), str(selector)))
        for varName in self.variables:
            variable =  dset( varName, selector ) if selector else dset( varName ) # type: cdms.tvariable.TransientVariable
            if dates is None:
                timeAxis = variable.getTime()      # type: cdms.axis.Axis
                dates = [ datetime.date() for datetime in timeAxis.asdatetime() ]
            timeseries =  variable.data  # type: np.ndarray
            if( length == -1 ): length = timeseries.shape[0] - offset
            offset_data = timeseries[offset:offset+length]
            norm_data = Analytics.normalize(offset_data) if normalize else offset_data
            for iS in range(smooth): norm_data = Analytics.lowpass(norm_data)
            norm_timeseries.append( (varName, norm_data ) )
        dset.close()
        return TimeseriesData( dates, norm_timeseries )

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()

    def deserialize(self):
        return None


class IITMDataSource(DataSource):

    # Types:   MONTHLY ANNUAL JF MAM JJAS OND
    # Names:   AI EPI NCI NEI NMI SPI WPI

    def __init__(self, name, _type):
        # type: ( str, str ) -> None
        DataSource.__init__(self, name)
        self.type = _type.lower()
        self.dataFile = self.getDataFilePath("IITM","txt")

    def serialize(self,lines):
        lines.append( "@IITMDataSource:" + ",".join( [ self.name, self.type ] ) )

    @staticmethod
    def deserialize( spec ):
        toks =  spec.split(":")[1].split(",")
        return IITMDataSource( toks[0], toks[1] )

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

    def freq(self):
        if self.type == "monthly": return "M"
        else: return "Y"

    def getTimeseries(self, timeRange, **kwargs):
        # type: ( CTimeRange ) -> TimeseriesData
        dset = open(self.dataFile,"r")
        lines = dset.readlines()
        normalize = kwargs.get("norm", True)
        smooth = kwargs.get("smooth", 0)
        decycle = kwargs.get( "decycle", False )

        timeseries = []
        dates = []
        logging.info("TrainingData: name = {0}, time range = {1}".format(str(self.name), str(timeRange)))
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
                    date = datetime.date(iYear, iMonth, 15)
                    if not timeRange or timeRange.inDateRange(date):
                        value = float( line_elems[ colIndex ] )
                        timeseries.append( value )
                        dates.append( date )
        tsarray = np.array( timeseries )

        decycled_array = Analytics.decycle( dates, tsarray ) if decycle else Analytics.center( tsarray )
        norm_data = Analytics.normalize(decycled_array) if normalize else decycled_array
        for iS in range(smooth): norm_data = Analytics.lowpass(norm_data)

        return TimeseriesData( dates, [ ( self.type, norm_data ) ] )

    def isYear( self, s ):
        # type: (str) -> bool
        try:
            test_val = int(s)
            return ( test_val > 1700 and test_val < 3000)
        except ValueError:
            return False

class TrainingDataset:

    def __init__(self, sources, **kwargs ):
        # type: ( list[DataSource], CDuration ) -> None
        self.dataSources = sources
        self.smooth = kwargs.get("smooth",0)
        self.decycle = kwargs.get("decycle", False)
        self.trainingRange = kwargs.get( "trainingRange", None ) # type: CTimeRange
        self._timeseries = []

    def serialize(self, lines ):
        # type: (list[str]) -> None
        lines.append( "#TrainingDataset" )
        Parser.sparm( lines, "smooth", self.smooth )
        Parser.sparm( lines, "decycle", self.decycle )
        if self.trainingRange:
            Parser.sparm( lines, "trainingRange", self.trainingRange.serialize() )
        for source in self.dataSources: source.serialize(lines)

    @staticmethod
    def deserialize( lines ):
        # type: (list[str]) -> TrainingDataset
        active = False
        parms = {}
        dataSources = []
        for line in lines:
            if line.startswith("#TrainingDataset"):
                active = True
            elif active:
                if line.startswith("#"): break
                elif line.startswith("@P:"):
                    toks = line.split(":")[1].split("=")
                    parms[toks[0]] = toks[1]
                elif line.startswith("@IITMDataSource:"):
                    dataSources.append( IITMDataSource.deserialize(line) )
        return TrainingDataset( dataSources, **parms )

    @classmethod
    def new(cls, sources, inputDataset, predictionLag, **kwargs):
        # type: ( list[DataSource], PCDataset, CDuration ) -> TrainingDataset
        offset = inputDataset.nTSteps - 1 if inputDataset is not None else 0
        smooth = inputDataset.smooth if inputDataset is not None else 0
        trainingRange = inputDataset.timeRange.shift(predictionLag.inc(offset))
 #       trainingRange1 = trainingRange.extend( CDuration.months( offset ) )
        return cls( sources, smooth=smooth, trainingRange=trainingRange, **kwargs )

    def targetNames(self):
        return [ source.name for source in self.dataSources ]

    def addDataSource(self, dsource ):
        # type: (DataSource) -> None
        self.dataSources.append( dsource )

    def getTimeseries(self):
        # type: () -> ( list[TimeseriesData] )
        if len( self._timeseries ) == 0:
            self._timeseries = [ dsource.getTimeseries( self.trainingRange, smooth=self.smooth, decycle=self.decycle ) for dsource in self.dataSources]  # type: List[TimeseriesData]
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