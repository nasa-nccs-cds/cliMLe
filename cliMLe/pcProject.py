import os, sys, math
import cdms2 as cdms
import logging, traceback
import numpy as np
from typing import List, Union
from cliMLe.dataProcessing import Analytics, CTimeRange, CDuration

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

    def getVariable( self, varName, rType, selector = None ):
        # type: (str, int) -> cdms.fvariable.FileVariable
        dataset = self.getDataset( rType )
        return dataset(varName,selector) if selector else dataset(varName)

    def getVariables( self, rType, selector = None ):
        # type: (int) -> list[cdms.tvariable.TransientVariable]
        dataset = self.getDataset( rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        logging.info( "Get Input PC timeseries, vars: {0}, range: {1}".format( str(varnames), str(selector) ) )
        return [ dataset(varName,selector) for varName in varnames ] if selector else [ dataset(varName) for varName in varnames ]


class PCDataset:

    def __init__(self, _experiments, **kwargs ):
       # type: ( list[Experiment] ) -> None
       self.timeRange = kwargs.get( "timeRange", None ) # type: CTimeRange
       self.normalize = kwargs.get("norm",True)
       self.nTSteps = kwargs.get( "nts", 1 )
       self.smooth = kwargs.get( "smooth", 0 )
       self.freq = kwargs.get("freq", "M")
       self.filter = kwargs.get("filter", "")
       self.experiments = _experiments if isinstance( _experiments, (list, tuple) ) else [ _experiments ] # type: List[Experiment]
       self.selector = self.timeRange.selector() if self.timeRange else None
       input_cols = []
       if self.filter :
           input_cols.extend( [self.preprocess(var) for var in self.getVariables()] )
           self.data = np.column_stack( input_cols )
           self.nTSteps = 1
       else:
           for iTS in range( self.nTSteps ):
               input_cols.extend( [self.preprocess(var,iTS) for var in self.getVariables()] )
           self.data = np.column_stack( input_cols )

    def getVariables(self):
        # type: () -> list[cdms.tvariable.TransientVariable]
        variable_list = []
        for experiment in self.experiments:
            variable_list.extend( experiment.getVariables( PC, self.selector ) )
        return variable_list

    def getInputDimension(self):
        # type: () -> int
        return self.data.shape[1]
#        size = 0
#        for experiment in self.experiments:
#            size += experiment.nModes
#        return size * self.nTSteps

    def getVariableIds(self):
        # type: () -> str
        return "[" + ",".join( exp.variable.id for exp in self.experiments ) +"]"

    def getFilterIndices(self):
        try:
            start_index = int(self.filter)
            return (start_index, 1)
        except:
            start_index = "xjfmamjjasondjfmamjjasond".index(self.filter.lower())
            if start_index < 0: raise Exception("Unrecognizable filter value: " + self.filter )
            return ( start_index, len(self.filter) )

    def preprocess(self, var, offset=0 ):
        end = var.shape[0] - (self.nTSteps-offset) + 1
        raw_data = var.data[offset:end]
        norm_data = Analytics.normalize(raw_data) if self.normalize else raw_data
        for iS in range( self.smooth ): norm_data = Analytics.lowpass(norm_data)
        if self.filter:
            slices = []
            dates = var.getTime().asComponentTime()
            months = [ date.month for date in dates ]
            (month_filter_start_index, filter_len) = self.getFilterIndices()
            for mIndex in range(len(months)):
                if months[mIndex] == month_filter_start_index:
                    slice = norm_data[ mIndex: mIndex + filter_len ]
                    slices.append( slice )
            batched_data = np.row_stack( slices )
            if self.freq == "M": batched_data = batched_data.flatten()
        else:
            batched_data = Analytics.yearlyAve( self.timeRange.startDate, "M", norm_data ) if self.freq == "Y" else norm_data

        return batched_data

    def getEpoch(self):
        # type: () -> np.ndarray
        return self.data