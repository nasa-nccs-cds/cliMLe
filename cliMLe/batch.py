import os, sys, math
import cdms2 as cdms
import numpy as np
from pcProject import *
from trainingData import *

class EpocServer:

    def __init__(self, _project, _experiment, _trainingData, _validataion_fraction, _normalize = True ):
       # type: (Project, str, TrainingDataset, int, int) -> BatchServer
       self.validataion_fraction = _validataion_fraction
       self.normalize = _normalize
       self.experiment = _experiment
       self.project = _project
       self.trainingData = _trainingData
       self.variables = _project.getVariables( self.experiment, PC ) # type: list[cdms.tvariable.TransientVariable]
       self.size = self.variables[0].shape[0]
       self.validationSize = math.ceil( self.size *  self.validataion_fraction )
       self.epocSize = self.size - self.validationSize
       self.inputData = np.column_stack( [ self.preprocess(var[:].data) for var in self.variables ] )
       self.outputData = np.column_stack( self.trainingData.getTimeseries() )
       self.output_size = self.outputData.shape[1]

    def preprocess(self, data ):
        if self.normalize:
            std = np.std( data, 0 )
            return data / std
        else:
            return data

    def getEpoch(self):
        return self.inputData, self.outputData

    def getTrain(self):
        return self.inputData[0:self.epocSize], self.outputData[0:self.epocSize]

    def getValidation(self):
        return self.inputData[self.epocSize:], self.outputData[self.epocSize:]

class BatchServer:

    def __init__(self, _project, _experiment, _trainingData, _batchSize = sys.maxint, _shuffleMode = NONE, _normalize = True ):
       # type: (Project, str, TrainingDataset, int, int) -> BatchServer
       self.batchSizeRequest = _batchSize
       self.shuffleMode = _shuffleMode
       self.normalize = _normalize
       self.experiment = _experiment
       self.project = _project
       self.trainingData = _trainingData
       self.batchIndex = -1
       self.variables = _project.getVariables( self.experiment, PC ) # type: list[cdms.tvariable.TransientVariable]
       self.size = self.variables[0].shape[0]
       self.numBatches = int( round( float( self.size ) / self.batchSizeRequest ) )
       self.batchSize = math.ceil( float( self.size ) / self.numBatches )
       data_series = [ self.preprocess(var[:].data) for var in self.variables ]
       training_series = self.trainingData.getTimeseries()
       self.input_size = len(data_series)
       self.output_size = len(training_series)
       data_series.extend( training_series )
       self.series = np.column_stack( data_series )
       self.perm_series = self.series
       self.currentBatch = None

    def preprocess(self, data ):
        if self.normalize:
            std = np.std( data, 0 )
            return data / std
        else:
            return data

    def getNextBatch(self):
        self.batchIndex = ( self.batchIndex + 1 ) % self.numBatches
        start_index = self.batchIndex * self.batchSize
        end_index = min( start_index + self.batchSize, self.size )
        if (self.shuffleMode == EPOCH) and (self.batchIndex == 0):
            self.perm_series = np.random.permutation(self.series)
        pc_batch = self.perm_series[start_index:end_index]
        perm_batch =  np.random.permutation(pc_batch) if (self.shuffleMode == BATCH) else pc_batch
        self.currentBatch = perm_batch[:,0:self.input_size], perm_batch[:,self.input_size:]
        return self.currentBatch

    def getCurrentBatch(self):
        return self.currentBatch

    def getEpoch(self):
        if self.shuffleMode == EPOCH:
            self.perm_series = np.random.permutation(self.series)
        return self.perm_series[:,0:self.input_size], self.perm_series[:,self.input_size:]

if __name__ == "__main__":
    projectName = "MERRA2_EOFs"
    varName = "ts"
    data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/Reanalysis/NASA-GMAO/GEOS-5/MERRA2/mon/atmos/' + varName + '.ncml'
    outDir = os.path.expanduser("~/results/")
    start_year = 1980
    end_year = 2015
    nModes = 20
    experiment = projectName + '_' + str(start_year) + '-' + str(end_year) + '_M' + str(nModes) + "_" + varName
    project = Project(outDir,projectName)

    td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], ( "1980-1-1", "2014-12-31" ) )
    dset = TrainingDataset( [td] )

    print
    print "SHUFFLE: BATCH"
    bserv = BatchServer( project, experiment, dset, 84, NONE )
    print "EPOC Size: " + str(  bserv.series.shape )
    print "NBatches: " + str(  bserv.numBatches )

    input, output = bserv.getEpoch()
    print "Epoch, input shape = " + str(input.shape) + ", output shape = " + str(output.shape)
    print "Input:"
    print str(input)
    print "Output:"
    print str(output)

    print "\nEPOC 1"
    for iBatch in range( bserv.numBatches ):
        input, output = bserv.getNextBatch()
        print "Batch " + str( iBatch ) + ", input shape = " + str( input.shape ) + ", output shape = " + str( output.shape )
        print "Input:"
        print str( input )
        print "Output:"
        print str( output )


    print "\nEPOC 2"
    for iBatch in range( bserv.numBatches ):
        input, output = bserv.getNextBatch()
        print "Batch " + str( iBatch ) + ", input shape = " + str( input.shape ) + ", output shape = " + str( output.shape )
        print "Input:"
        print str( input )
        print "Output:"
        print str( output )






