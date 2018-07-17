import abc, os, logging
import numpy as np
import cdms2 as cdms
from cliMLe.dataProcessing import Analytics, CTimeRange, CDuration

class InputDataSource:
    __metaclass__ = abc.ABCMeta

    def __init__(self, _name, **kwargs ):
       self.name = _name
       self.timeRange = kwargs.get( "timeRange", None ) # type: CTimeRange
       self.normalize = kwargs.get("norm",True)
       self.nTSteps = kwargs.get( "nts", 1 )
       self.smooth = kwargs.get( "smooth", 0 )
       self.decycle = kwargs.get("decycle", False)
       self.freq = kwargs.get("freq", "M")
       self.filter = kwargs.get("filter", "")
       self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def getDataFilePath(self, type, ext ):
        # type: (str,str) -> str
        return os.path.join(os.path.join(self.rootDir, "data",type), self.name + "." + ext )

    def getName(self):
        return self.name

    @abc.abstractmethod
    def getEpoch(self):
        # type: () -> np.ndarray
        return

    @abc.abstractmethod
    def getInputDimension(self):
        # type: () -> int
        return


class InputDataset:

    def __init__(self, _sources ):
       # type: (list[InputDataSource]) -> None
       self.sources = _sources

    def getName(self):
        return "-".join( [ source.getName() for source in self.sources ] )

    def getEpoch(self):
        # type: () -> np.ndarray
        epocs = [ source.getEpoch() for source in self.sources ]
        result = np.column_stack( epocs )
        return result

    def getInputDimension(self):
        # type: () -> int
        return sum( [ source.getInputDimension() for source in self.sources ] )


class CDMSInputSource(InputDataSource):

    def __init__(self, name, variableList, **kwargs ):
        InputDataSource.__init__( self, name, **kwargs )
        self.variables = variableList
        self.dataFile = self.getDataFilePath("cvdp","nc")
        self.variables = variableList
        self.data = None

    def getTimeseries( self ):
        # type: () -> list[np.ndarray]
        debug = False
        dset = cdms.open( self.dataFile )
        norm_timeseries = []
        dates = None
        selector = self.timeRange.selector() if self.timeRange else None
        logging.info( "InputData: variables = {0}, time range = {1}".format( str(self.variables), str(selector)))
        for varName in self.variables:
            variable =  dset( varName, selector ) if selector else dset( varName ) # type: cdms.tvariable.TransientVariable
            if dates is None:
                timeAxis = variable.getTime()      # type: cdms.axis.Axis
                dates = [ datetime.date() for datetime in timeAxis.asdatetime() ]
            timeseries =  variable.data  # type: np.ndarray
            if debug:
                logging.info( "-------------- RAW DATA ------------------ " )
                for index in range( len(dates)):
                    logging.info( str(dates[index]) + ": " + str( timeseries[index] ) )
            if self.filter:
                slices = []
                months = [ date.month for date in dates ]
                (month_filter_start_index, filter_len) = Analytics.getMonthFilterIndices(self.filter)
                for mIndex in range(len(months)):
                    if months[mIndex] == month_filter_start_index:
                        slice = timeseries[ mIndex: mIndex + filter_len ]
                        slices.append( slice )
                batched_data = np.row_stack( slices )
                if self.freq == "M": batched_data = batched_data.flatten()
            else:
                batched_data = Analytics.yearlyAve( self.timeRange.startDate, "M", timeseries ) if self.freq == "Y" else timeseries
            if debug:
                logging.info( "-------------- FILTERED DATA ------------------ " )
                for index in range( batched_data.shape[0]):
                    logging.info( str( batched_data[index] ) )
            norm_data = Analytics.normalize(batched_data) if self.normalize else batched_data
            for iS in range(self.smooth): norm_data = Analytics.lowpass(norm_data)
            norm_timeseries.append( norm_data )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()

    def initialize(self):
        if self.data == None:
            timeseries = self.getTimeseries()
            self.data = np.column_stack( timeseries )

    def getEpoch(self):
        # type: () -> np.ndarray
        self.initialize()
        return self.data

    def getInputDimension(self):
        # type: () -> int
        self.initialize()
        return self.data.shape[1]
