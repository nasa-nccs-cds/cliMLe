import os, itertools
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector
import matplotlib.pyplot as plt

class DataSource:

    def __init__(self, variableList, time_bounds ):
        self.variables = variableList
        self.time_bounds = time_bounds
        self.selector = Selector(time=self.time_bounds )

    def getTimeseries(self): raise NotImplementedError
        # type: (bool) -> list[np.ndarray]


class ProjectDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds ):
        DataSource.__init__( self, variableList, time_bounds )
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.dataFile = os.path.join( os.path.join( self.rootDir, "data" ), name + ".nc" )
        self.variables = variableList

    def getTimeseries(self, normalize = True ):
        # type: (bool) -> list[np.ndarray]
        dset = cdms.open( self.dataFile )
        norm_timeseries = []
        for varName in self.variables:
            variable =  dset.getVariable( varName )
            timeseries =  variable(self.selector).data
            if normalize:
                std = np.std( timeseries, 0 )
                norm_timeseries.append( timeseries / std )
            else:
                norm_timeseries.append( timeseries )
        dset.close()
        return norm_timeseries

    def listVariables(self):
        # type: () -> list[str]
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()

class TrainingDataset:

    def __init__(self, sources = [] ):
        # type: (list[DataSource]) -> None
        self.dataSources = sources

    def addDataSource(self, dsource ):
        self.dataSources.append( dsource )

    def getTimeseries(self):
        timeseries = []
        for dsource in self.dataSources: timeseries.extend( dsource.getTimeseries() )
        return timeseries

    def getTimeseriesData(self):
        # type: () -> np.ndarray
        timeseries = []
        for dsource in self.dataSources: timeseries.extend( dsource.getTimeseries() )
        return np.column_stack( timeseries )

    def plotTimeseries(self):
        plt.title("Training data")
        for (tsname,tsdata) in self.getTimeseries().items():
            plt.plot( tsdata, label = tsname )
        plt.legend()
        plt.show()
        
if __name__ == "__main__":

    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )
    print "Variables: " + str( td.listVariables() )
    dset = TrainingDataset( [td] )
    dset.plotTimeseries()