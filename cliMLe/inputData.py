import abc, os, logging, cdtime, time
import numpy as np
import cdms2 as cdms
from cdms2.selectors import Selector
from cdms2 import level
from typing import Optional, Any

from cliMLe.dataProcessing import Analytics, CTimeRange, Parser
from cliMLe.project import *
from cliMLe.svd import Eof

# NONE = 0
# BATCH = 1
# EPOCH = 2

class InputDataSource(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, _name, **kwargs ):
       self.name = _name
       self.timeRange = CTimeRange.deserialize( kwargs.get( "timeRange", None ) ) # type: CTimeRange
       self.normalize = bool( kwargs.get("norm","True") )
       self.nTSteps = int( kwargs.get( "nts", "1" ) )
       self.smooth = int( kwargs.get( "smooth", "0" ) )
       self.decycle = bool( kwargs.get("decycle", "False" ) )
       self.freq = kwargs.get("freq", "Y")
       self.filter = kwargs.get("filter", "")
       self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
       self.data = None

    def serialize(self,lines):
        Parser.sparm( lines, "normalize", self.normalize )
        Parser.sparm( lines, "nTSteps", self.nTSteps )
        Parser.sparm( lines, "smooth", self.smooth )
        Parser.sparm( lines, "decycle", self.decycle )
        Parser.sparm( lines, "freq", self.freq )
        Parser.sparm( lines, "filter", self.filter )
        Parser.sparm( lines, "timeRange", self.timeRange.serialize() )

    def getEOFWeights(self, nModes, **kwargs):
        self.normalize = True
        self.initialize()
        self.solver = Eof( self.data, **kwargs )
        eofs = self.solver.eofs( neofs=nModes )
        self.pcs = self.solver.pcs( npcs=nModes ).transpose()
#        self.projected_pcs = self.solver.projectField( self.data,neofs=nModes).transpose()
        self.projected_pcs = np.dot( self.data, eofs.T )
        weight_norm = self.projected_pcs.std(0).reshape([-1,1])
        self.eofWeights = eofs / weight_norm
        return self.eofWeights

    def norm(self,data,axis=0):
        # type: (np.ndarray) -> np.ndarray
        mean = data.mean(axis)
        centered = (data - mean)
        std = centered.std(axis)
        return centered / std

    def getDataFilePath(self, type, ext ):
        # type: (str,str) -> str
        return os.path.join(os.path.join(self.rootDir, "data",type), self.name + "." + ext )

    def getName(self):
        return self.name

    def getEpoch(self):
        # type: () -> np.ndarray
        self.initialize()
        return self.data

    @abc.abstractmethod
    def initialize(self):
        # type: () -> None
        return

    def getInputDimension(self):
        # type: () -> int
        self.initialize()
        return self.data.shape[1]

    @abc.abstractmethod
    def getVariableIds(self):
        # type: () -> str
        return

    @abc.abstractmethod
    def getExperimentIds(self):
        # type: () -> str
        return

class InputDataset(object):

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

    def getEOFWeights(self, nModes, **kwargs):
        # type: () -> np.ndarray
        pcs = [source.getEOFWeights(nModes, **kwargs) for source in self.sources]
        result = np.column_stack( pcs )
        return result.T

    def getInputDimension(self):
        # type: () -> int
        return sum( [ source.getInputDimension() for source in self.sources ] )

    def serialize(self, lines ):
        for source in self.sources: source.serialize(lines)

    def getVariableIds(self):
        return "--". join( [ src.getVariableIds() for src in self.sources ] )

    def getExperimentIds(self):
        return "--". join( [ src.getExperimentIds() for src in self.sources ] )

    @staticmethod
    def deserialize( lines ):
        # type: (list[str]) -> InputDataset
        from cliMLe.climatele import ClimateleDataset
        sources = []
        for iLine in range(len(lines)):
            if lines[iLine].startswith("#PCDataset"):
                sources.append(ClimateleDataset.deserialize(lines[iLine:]))
        return InputDataset( sources )

class CDMSInputSource(InputDataSource):

    def __init__(self, name, variableList, **kwargs ):
        super(CDMSInputSource, self).__init__(name, **kwargs)
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
                for mIndex in range(len(

                )):
                    if months[mIndex] == month_filter_start_index:
                        slice = timeseries[ mIndex: mIndex + filter_len ]
                        slices.append( slice )
                batched_data = np.row_stack( slices )
                if self.freq == "M": batched_data = batched_data.flatten()
                elif self.freq == "YA": batched_data = np.average(batched_data,1)
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

    def getVariableIds(self):
        # type: () -> str
        return "-".join( self.listVariables() )

    @abc.abstractmethod
    def getExperimentIds(self):
        return ""


    def initialize(self):
        if self.data == None:
            timeseries = self.getTimeseries()
            self.data = np.column_stack( timeseries )


class ReanalysisInputSource(InputDataSource):

    def __init__(self, name, variableList, **kwargs ):
        # type: (str, list[ODAPVariable] ) -> None
        super(ReanalysisInputSource, self).__init__(name, **kwargs)
        self.parms = kwargs
        self.variables = variableList
        self.timeRange = self.parms.get("time",None)  # type: Optional[CTimeRange]
        if self.timeRange: assert isinstance(self.timeRange, CTimeRange), "Time bounds must be instance of CTimeRange, found: " + self.timeRange.__class__.__name__
        self.data = None

    def getSelector(self):
        selector = Selector()
        level_val = self.parms.get("level",None)
        if level:
            selector = selector & level(level_val)
        lat_bounds = self.parms.get("latitude",None)
        if lat_bounds:
            selector = selector & Selector(latitude=lat_bounds)
        lon_bounds = self.parms.get("longitude",None)
        if lon_bounds:
            selector = selector & Selector(longitude=lon_bounds)
        if self.timeRange:
            selector = selector & self.timeRange.selector()
        return selector

    def getTimeseries( self ):
        # type: () -> list[np.ndarray]
        debug = False
        norm_timeseries = []
        dates = None
        roi_selector = self.getSelector()
        for var in self.variables:
            dset = cdms.open( var.dsetUrl )
            variable =  dset( var.varname, var.getSelector( roi_selector ) ).squeeze() # type: cdms.tvariable.TransientVariable
            if dates is None:
                timeAxis = variable.getTime()      # type: cdms.axis.Axis
                dates = [ datetime.date() for datetime in timeAxis.asdatetime() ]
            timeseries =  variable.data  # type: np.ndarray
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
                elif self.freq == "YA": batched_data = np.average(batched_data,1)
            else:
                batched_data = Analytics.yearlyAve( self.timeRange.startDate, "M", timeseries ) if self.freq == "Y" else timeseries
            batched_data = batched_data.reshape( [ batched_data.shape[0], -1 ] )
            if self.normalize:
                centered_data = Analytics.center( batched_data )
                batched_data = Analytics.normalize( centered_data )
            for iS in range(self.smooth): batched_data = Analytics.lowpass( batched_data )
            norm_timeseries.append( batched_data )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        return [ var.varname for var in self.variables ]

    def getVariableIds(self):
        # type: () -> str
        return "-".join( self.listVariables() )

    def initialize(self):
        if self.data is None:
            timeseries = self.getTimeseries()
            self.data = np.column_stack( timeseries )

    def getExperimentIds(self):
        return []

if __name__ == "__main__":

    def getDsetUrl( pname, varname ):
        if pname.lower() == "merra2": data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/Reanalysis/NASA-GMAO/GEOS-5/MERRA2/mon/atmos/' + varname + '.ncml'
        elif pname.lower() == "20crv2c": data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/reanalysis/20CRv2c/mon/atmos/' + varname + '.ncml'
        else: raise Exception( " Unrecognized project: " + pname )
        return data_path

    projectName = "merra2"
    variables = [ODAPVariable("ts", getDsetUrl(projectName, "ts"))]
    start_year = 1980
    end_year = 2015
    start_time = cdtime.comptime(start_year)
    end_time = cdtime.comptime(end_year)
    filter = "8"
    freq="Y"

    inputs = ReanalysisInputSource(projectName, variables, latitude=(-80,80), time=(start_time,end_time), filter=filter, freq=freq )

    epoch = inputs.getEpoch()

    print "Epoch shape = " + str( epoch.shape )
    print "Epoch sample = " + str( epoch[0] )



