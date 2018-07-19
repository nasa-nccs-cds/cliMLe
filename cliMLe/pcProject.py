import os, sys, math
import cdms2 as cdms
import logging, traceback
import numpy as np
from typing import List, Union
from cliMLe.dataProcessing import Analytics, CTimeRange, Parser
from cliMLe.inputData import InputDataSource

PC = 0
EOF = 1

NONE = 0
BATCH = 1
EPOCH = 2

class Variable:
    def __init__(self, _varname, _level = None ):
        self.varname = _varname
        self.level = Variable.deserialize( _level )
        self.id = self.varname + "-" + str(self.level) if self.level else self.varname

    @staticmethod
    def deserialize( spec ):
        if spec is None: return None
        elif isinstance( spec, str ):
            spec = spec.lower().strip()
            if spec == "none": return None
            else: return int( spec )
        else: return spec


class Project:

    def __init__(self, _dir, _name ):
        self.name = _name
        self.directory = _dir
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    @staticmethod
    def new( baseDir, name ):
        directory = os.path.join( baseDir, name )
        return Project( directory, name )

class Experiment:

    def __init__(self, _project, _start_year, _end_year, _nModes, _variable ):
        self.exts = { PC: '-PCs.nc', EOF: '-EOFs.nc' }
        self.project = _project
        self.start_year = _start_year
        self.end_year = _end_year
        self.nModes = _nModes
        self.variable = _variable
        self.id = self.project.name + '_' + str(self.start_year) + '-' + str(self.end_year) + '_M' + str(self.nModes) + "_" + self.variable.id

    def serialize( self, lines ):
        lines.append( "@E:" + ",".join( [ self.project.directory, self.project.name, str(self.start_year), str(self.end_year), str(self.nModes), self.variable.varname, str(self.variable.level) ] ) )

    @staticmethod
    def deserialize( spec ):
        # type: (str) -> Experiment
        toks = spec.split(":")[1].split(",")
        return Experiment( Project( toks[0], toks[1] ),  int(toks[2]), int(toks[3]), int(toks[4]), Variable( toks[5], toks[6] ) )

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

    def getVariables( self, rType, selector = None, nModes = -1 ):
        # type: (int) -> list[cdms.tvariable.TransientVariable]
        dataset = self.getDataset( rType )
        varnames = [ vname for vname in dataset.listvariables() if not vname.endswith("_bnds") ]
        varnames.sort()
        nVars = len(varnames) if nModes <0 else min( len(varnames), nModes )
        logging.info( "Get Input PC timeseries, vars: {0}, range: {1}".format( str(varnames), str(selector) ) )
        return [ dataset(varnames[iVar],selector) for iVar in range(nVars) ] if selector else [ dataset( varnames[iVar] ) for iVar in range(nVars) ]


class PCDataset(InputDataSource):

    def __init__(self, name, _experiments, **kwargs ):
       # type: ( str, list[Experiment] ) -> None
       super(PCDataset, self).__init__(name, **kwargs)
       self.nModes = int( kwargs.get( "nmodes", "-1" ) )
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

    def serialize(self,lines):
        lines.append( "#PCDataset:" + self.name )
        Parser.sparm( lines, "nmodes", self.nModes )
        InputDataSource.serialize(self,lines)
        for experiment in self.experiments: experiment.serialize(lines)

    @staticmethod
    def deserialize( lines ):
        # type: (list[str]) -> PCDataset
        active = False
        dsname = None
        parms = {}
        experiments = []
        for line in lines:
            if line.startswith("#PCDataset"):
                active = True
                dsname = line.split(":")[1]
            elif active:
                if line.startswith("#"): break
                elif line.startswith("@P:"):
                    toks = line.split(":")[1].split("=")
                    parms[toks[0]] = toks[1].strip()
                elif line.startswith("@E:"):
                    experiments.append( Experiment.deserialize(line) )
        return PCDataset( dsname, experiments, **parms )

    def getVariables(self):
        # type: () -> list[cdms.tvariable.TransientVariable]
        variable_list = []
        for experiment in self.experiments:
            variable_list.extend( experiment.getVariables( PC, self.selector, self.nModes ) )
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

    def getExperimentIds(self):
        return "-". join( [ exp.id for exp in self.experiments ] )

    def preprocess(self, var, offset=0 ):
        debug = False
        end = var.shape[0] - (self.nTSteps-offset) + 1
        raw_data = var.data[ offset:end ]
        norm_data = Analytics.normalize(raw_data) if self.normalize else raw_data
        for iS in range( self.smooth ): norm_data = Analytics.lowpass(norm_data)

        if self.filter:
            slices = []
            dates = var.getTime().asComponentTime()
            if debug:
                logging.info( "-------------- RAW DATA ------------------ " )
                for index in range( len(dates)):
                    logging.info( str(dates[index]) + ": " + str( norm_data[index] ) )

            months = [ date.month for date in dates ]
            (month_filter_start_index, filter_len) = Analytics.getMonthFilterIndices(self.filter)
            for mIndex in range(len(months)):
                if months[mIndex] == month_filter_start_index:
                    slice = norm_data[ mIndex: mIndex + filter_len ]
                    slices.append( slice )
            batched_data = np.row_stack( slices )
            if self.freq == "M": batched_data = batched_data.flatten()

            if debug:
                logging.info( "-------------- FILTERED DATA ------------------ " )
                for index in range( batched_data.shape[0]):
                    logging.info( str( batched_data[index] ) )
        else:
            batched_data = Analytics.yearlyAve( self.timeRange.startDate, "M", norm_data ) if self.freq == "Y" else norm_data

        return batched_data

    def getEpoch(self):
        # type: () -> np.ndarray
        return self.data