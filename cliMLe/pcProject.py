import os, sys, math
import cdms2 as cdms
import numpy as np
from typing import List, Union

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
        # type: (int) -> dict[str,cdms.tvariable.TransientVariable]
        dataset = self.getDataset( rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        return { varName: dataset(varName) for varName in varnames }

    def getVariableData( self, rType ):
        # type: (int) -> dict[str,np.ndarray]
        dataset = self.getDataset( rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        return { varName: dataset(varName).data for varName in varnames }


class PCDataset:

    def __init__(self, _experiments ):
        self.experiments = _experiments if isinstance( _experiments, (list, tuple) ) else [ _experiments ] # type: List[Experiment]

    def getVariables(self):
        # type: () -> dict[str,cdms.tvariable.TransientVariable]
        variable_map = {}
        for experiment in self.experiments:
            variable_map.update( experiment.getVariables( PC ) )
        return variable_map

    def getVariableData(self):
        # type: () -> np.ndarray
        variable_list = []
        for experiment in self.experiments:
            variable_list.extend( experiment.getVariableData( PC ).values() )
        return np.column_stack( variable_list )

    def getInputDimension(self):
        # type: () -> int
        size = 0
        for experiment in self.experiments:
            size += experiment.nModes
        return size

    def getVariableIds(self):
        # type: () -> str
        return "[" + ",".join( exp.variable.id for exp in self.experiments ) +"]"