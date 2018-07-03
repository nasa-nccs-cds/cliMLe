import os, itertools
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector
import matplotlib.pyplot as plt
from cliMLe.dataProcessing import PreProc

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
            smoothed_data = PreProc.lowpass( norm_data, smooth ) if smooth else norm_data
            norm_timeseries.append( (varName, smoothed_data ) )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()

class TrainingDataset:

    def __init__(self, sources = [], **kwargs ):
        # type: (list[DataSource]) -> None
        self.dataSources = sources
        self.offset = kwargs.get("offset",0)
        self.smooth = kwargs.get("smooth",0)
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