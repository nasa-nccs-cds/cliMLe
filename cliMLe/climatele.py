import os, sys, math
import cdms2 as cdms
import logging, traceback
import numpy as np
from typing import List, Union
from cliMLe.dataProcessing import Analytics, CTimeRange, Parser
from cliMLe.inputData import InputDataSource
from cliMLe.project import Project, Experiment, PC, EOF

class ClimateleDataset(InputDataSource):

    def __init__(self, name, _experiments, **kwargs ):
       # type: ( str, list[Experiment] ) -> None
       super(ClimateleDataset, self).__init__(name, **kwargs)
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
        # type: (list[str]) -> ClimateleDataset
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
        return ClimateleDataset(dsname, experiments, **parms)

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
            if self.freq == "M":    batched_data = batched_data.flatten()
            elif self.freq == "YA": batched_data = np.average(batched_data,1)

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