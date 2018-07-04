import os, sys, math
import cdms2 as cdms
import numpy as np
from typing import List, Union
from cliMLe.dataProcessing import PreProc

PC = 0
EOF = 1

NONE = 0
BATCH = 1
EPOCH = 2

class Variable:
    def __init__(self, _varname, _level = None ):
        self.varname = _varname
        self.level = _level
        self.id = self.varname + "-" + str(self.level) if self.level else self.varname

class Project:

    def __init__(self, baseDir, _name ):
        self.name = _name
        self.directory = os.path.join( baseDir, _name )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

class Experiment:

    def __init__(self, _project, _start_year, _end_year, _nModes, _variable ):
        self.exts = { PC: '-PCs.nc', EOF: '-EOFs.nc' }
        self.project = _project
        self.start_year = _start_year
        self.end_year = _end_year
        self.nModes = _nModes
        self.variable = _variable
        self.id = self.project.name + '_' + str(self.start_year) + '-' + str(self.end_year) + '_M' + str(self.nModes) + "_" + self.variable.id

    def outfilePath(self, rType ):
        # type: (int) -> str
        return os.path.join( self.project.directory, self.id + self.exts.get( rType, "Invalid response type: " + str(rType) ) )

    def getVariableNames(self, rType ):
        # type: (int) -> list[str]
        dataset = self.getDataset( rType )
        return [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]

    def getDataset(self, rType ):
        # type: (int) -> cdms.dataset.CdmsFile
        filePath = self.outfilePath( rType )
        return cdms.open(filePath)

    def getVariable( self, varName, rType ):
        # type: (str, int) -> cdms.fvariable.FileVariable
        dataset = self.getDataset( rType )
        return dataset(varName)

    def getVariables( self, rType ):
        # type: (int) -> list[cdms.tvariable.TransientVariable]
        dataset = self.getDataset( rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        return [ dataset(varName) for varName in varnames ]


class PCDataset:

    def __init__(self, _experiments, **kwargs ):
       self.normalize = kwargs.get("norm",True)
       self.nTSteps = kwargs.get( "nts", 1 )
       self.smooth = kwargs.get( "smooth", 0 )
       self.experiments = _experiments if isinstance( _experiments, (list, tuple) ) else [ _experiments ] # type: List[Experiment]
       input_cols = []
       for iTS in range( self.nTSteps ):
           input_cols.extend( [self.preprocess(var,iTS) for var in self.getVariables()] )
       self.data = np.column_stack( input_cols )

    def getVariables(self):
        # type: () -> list[cdms.tvariable.TransientVariable]
        variable_list = []
        for experiment in self.experiments:
            variable_list.extend( experiment.getVariables( PC ) )
        return variable_list

    def getInputDimension(self):
        # type: () -> int
        size = 0
        for experiment in self.experiments:
            size += experiment.nModes
        return size * self.nTSteps

    def getVariableIds(self):
        # type: () -> str
        return "[" + ",".join( exp.variable.id for exp in self.experiments ) +"]"

    def preprocess(self, var, offset=0 ):
        end = var.shape[0] - (self.nTSteps-offset) + 1
        data = var.data[offset:end]
        norm_data = PreProc.normalize( data ) if self.normalize else data
        for iS in range( self.smooth ): norm_data = PreProc.lowpass( norm_data )
        return norm_data

    def getEpoch(self):
        # type: () -> np.ndarray
        return self.data