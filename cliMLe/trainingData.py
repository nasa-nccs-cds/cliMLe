import os, itertools
import cdms2 as cdms
import numpy as np
from cdms2.selectors import Selector

class DataSource:

    def __init__(self, variableList, time_bounds ):
        self.variables = variableList
        self.time_bounds = time_bounds
        self.selector = Selector(time=self.time_bounds )

    def getTimeseies(self): raise NotImplementedError


class ProjectDataSource(DataSource):

    def __init__(self, name, variableList, time_bounds ):
        DataSource.__init__( self, variableList, time_bounds )
        self.rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.dataFile = os.path.join( os.path.join( self.rootDir, "data" ), name + ".nc" )
        self.variables = variableList

    def getTimeseries(self):
        dset = cdms.open( self.dataFile )
        variables = [  dset.getVariable( varName ) for varName in self.variables ]
        timeseries = [ variable(self.selector).data for variable in variables ]
        dset.close()
        return timeseries

    def listVariables(self):
        dset = cdms.open( self.dataFile )
        return dset.variables.keys()

class TrainingDataset:

    def __init__(self, sources = [] ):
        self.dataSources = sources

    def addDataSource(self, dsource ):
        self.dataSources.append( dsource )

    def getTimeseries(self):
        timeseries = []
        for dsource in self.dataSources: timeseries.extend( dsource.getTimeseries() )
        return timeseries

if __name__ == "__main__":

    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )

    print "Variables: " + str( td.listVariables() )

    dset = TrainingDataset( [td] )

    ts = dset.getTimeseries()

    print "SHAPE: " + str( ts[0].shape )