import os, sys, math
import cdms2 as cdms
import numpy as np

PC = 0
EOF = 1

NONE = 0
BATCH = 1
SERIES = 2

class Project:

    def __init__(self, baseDir, _name ):
        self.name = _name

        self.exts = { PC: '-PCs.nc', EOF: '-EOFs.nc' }
        self.directory = os.path.join( baseDir, _name )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def outfilePath(self, experiment, rType ):
        # type: (str, int) -> str
        return os.path.join( self.directory, experiment + self.exts.get( rType, "Invalid response type: " + str(rType) ) )

    def getVariableNames(self, experiment, rType ):
        # type: (str, int) -> list[str]
        dataset = self.getDataset( experiment, rType )
        return [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]

    def getDataset(self, experiment, rType ):
        # type: (str, int) -> cdms.dataset.CdmsFile
         filePath = self.outfilePath( experiment, rType )
         return cdms.open(filePath)

    def getVariable( self, varName, experiment, rType ):
        # type: (str, str, int) -> cdms.fvariable.FileVariable
        dataset = self.getDataset( experiment, rType )
        return dataset(varName)

    def getVariables( self, experiment, rType ):
        # type: (str, int) -> list[cdms.tvariable.TransientVariable]
        dataset = self.getDataset( experiment, rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        return [ dataset(varName)[0:9] for varName in varnames ]


class BatchServer:

    def __init__(self, _project, _experiment, _batchSize = sys.maxint, _shuffleMode = NONE ):
       self.batchSizeRequest = _batchSize
       self.shuffleMode = _shuffleMode
       self.experiment = _experiment
       self.project = _project
       self.batchIndex = -1
       self.variables = _project.getVariables( self.experiment, PC ) # type: list[cdms.tvariable.TransientVariable]
       self.size = self.variables[0].shape[0]
       self.numBatches = int( round( float( self.size ) / self.batchSizeRequest ) )
       self.batchSize = math.ceil( float( self.size ) / self.numBatches )
       self.series = np.column_stack( [ var[:].data for var in self.variables ] )
       self.perm_series = self.series

    def getNextBatch(self):
        self.batchIndex = ( self.batchIndex + 1 ) % self.numBatches
        start_index = self.batchIndex * self.batchSize
        end_index = min( start_index + self.batchSize, self.size )
        if (self.shuffleMode == SERIES) and (self.batchIndex == 0):
            self.perm_series = np.random.permutation(self.series)
        pc_batch = self.perm_series[start_index:end_index]
        rv =  np.random.permutation(pc_batch) if (self.shuffleMode == BATCH) else pc_batch
        return rv

if __name__ == "__main__":
    projectName = "MERRA2_EOFs"
    outDir = "/tmp/"
    start_year = 1980
    end_year = 2000
    varName = "ts"
    experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_' + varName

    project = Project(outDir,projectName)

    print
    print "SHUFFLE: BATCH:"
    bserv = BatchServer( project, experiment, 3, BATCH )
    for iBatch in range(6):
        batche = bserv.getNextBatch()
        print "Batch " + str( iBatch ) + ":"
        print str( batche )






