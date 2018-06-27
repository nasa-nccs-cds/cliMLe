import os, sys, math
import cdms2 as cdms
import numpy as np
from pcProject import *
from trainingData import *

class EpocServer:

    def __init__(self, _pcDataset, _trainingData, _validataion_fraction, _normalize = True ):
       # type: (PCDataset, TrainingDataset, float, bool) -> EpocServer
       self.validataion_fraction = _validataion_fraction
       self.normalize = _normalize
       self.pcDataset = _pcDataset
       self.trainingData = _trainingData
       self.variables = self.pcDataset.getVariables()
       self.size = self.variables[0].shape[0]
       self.validationSize = math.ceil( self.size *  self.validataion_fraction )
       self.epocSize = self.size - self.validationSize
       self.inputData = np.column_stack( [ self.preprocess(var[:].data) for var in self.variables ] )
       self.outputData = np.column_stack( [ tsdata for (tsname,tsdata) in self.trainingData.getTimeseries() ] )
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