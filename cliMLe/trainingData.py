import os, itertools
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector
import matplotlib.pyplot as plt
from cliMLe.dataProcessing import PreProc
from cliMLe.pcProject import PCDataset

class DataSource:

    def __init__(self, variableList, time_bounds ):
        self.variables = variableList
        self.time_bounds = time_bounds
        self.selector = Selector(time=self.time_bounds )

    def getTimeseries(self): raise NotImplementedError


class ProjectDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds ):
        DataSource.__init__( self, variableList, time_bounds )
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.dataFile = os.path.join( os.path.join( self.rootDir, "data" ), name + ".nc" )
        self.variables = variableList

    def getTimeseries( self, **kwargs ):
        # type: (bool) -> list[(str,np.ndarray)]
        dset = cdms.open( self.dataFile )
        normalize = kwargs.get( "norm", True )
        smooth = kwargs.get( "smooth", 0 )
        offset = kwargs.get( "offset", 0 )
        length = kwargs.get( "length", -1 )
        norm_timeseries = []
        for varName in self.variables:
            variable =  dset.getVariable( varName )
            timeseries =  variable(self.selector).data  # type: np.ndarray
            if( length == -1 ): length = timeseries.shape[0] - offset
            offset_data = timeseries[offset:offset+length]
            norm_data = PreProc.normalize( offset_data ) if normalize else offset_data
            for iS in range(smooth): norm_data = PreProc.lowpass( norm_data )
            norm_timeseries.append( (varName, norm_data ) )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()


class IITMDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds):
        DataSource.__init__(self, variableList, time_bounds)
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.dataFile = os.path.join(os.path.join(self.rootDir, "data","IITM"), name + ".txt")
        self.variables = variableList
        self.varList = None

    def getTimeseries(self, **kwargs):
        # type: (bool) -> list[(str,np.ndarray)]
        dset = open(self.dataFile,"r")
        lines = dset.readlines()
        normalize = kwargs.get("norm", True)
        smooth = kwargs.get("smooth", 0)
        offset = kwargs.get("offset", 0)
        length = kwargs.get("length", -1)
        norm_timeseries = {}
        for iLine in range( len(lines) ):
            line_elems = lines[iLine].split()
            if line_elems[0] == "YEAR":
                self.varList = line_elems
            elif self.isYear(line_elems[0]):
                for varName in self.variables:
                    varIndex = self.varList.index( varName )
                    value = float( line_elems[ varIndex ] )
                    timeseries = norm_timeseries.get( varName, [] )
                    timeseries.append( value )

        # for varName in self.variables:
        #     variable = dset.getVariable(varName)
        #     timeseries = variable(self.selector).data  # type: np.ndarray
        #     if (length == -1): length = timeseries.shape[0] - offset
        #     offset_data = timeseries[offset:offset + length]
        #     norm_data = PreProc.normalize(offset_data) if normalize else offset_data
        #     for iS in range(smooth): norm_data = PreProc.lowpass(norm_data)
        #     norm_timeseries.append((varName, norm_data))
        # dset.close()
        # return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        return self.varList[1:]

    def isYear( self, s ):
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
        self.data = np.column_stack([tsdata for (tsname, tsdata) in self.getTimeseries()])
        self.output_size = self.data.shape[1]

    def addDataSource(self, dsource ):
        self.dataSources.append( dsource )

    def getTimeseries(self):
        # type: (bool) -> list[(str,np.ndarray)]
        timeseries = []
        for dsource in self.dataSources: timeseries.extend( dsource.getTimeseries( offset=self.offset, smooth=self.smooth ) )
        return timeseries

    def plotTimeseries(self):
        plt.title("Training data")
        for dsource in self.dataSources: 
            for (tsname,tsdata) in dsource.getTimeseries( offset=self.offset, smooth=self.smooth ):
                plt.plot( tsdata, label = tsname )
        plt.legend()
        plt.show()

    def getEpoch(self):
        # type: () -> np.ndarray
        return self.data

if __name__ == "__main__":

    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )
    print "Variables: " + str( td.listVariables() )
    dset = TrainingDataset( [td] )
    dset.plotTimeseries()