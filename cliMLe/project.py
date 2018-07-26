import abc, os, logging, cdtime, time
import cdms2 as cdms
from cdms2.selectors import Selector
from cdms2 import level

PC = 0
EOF = 1

class Variable(object):
    def __init__(self, _varname, _level = None ):
        self.varname = _varname
        self.level = Variable.rint(_level)
        self.id = self.varname + "-" + str(self.level) if self.level else self.varname

    def getSelector( self, selector ):
        return selector if self.level == None else selector & level( self.level )

    @staticmethod
    def rint(spec):
        if spec is None: return None
        elif isinstance( spec, str ):
            spec = spec.lower().strip()
            if spec == "none": return None
            else: return int( spec )
        else: return spec

class ODAPVariable(Variable):

    def __init__(self, _varname, _dsetUrl, _level = None ):
        super(ODAPVariable, self).__init__(_varname, _level)
        self.dsetUrl = _dsetUrl

class Project(object):

    def __init__(self, _dir, _name ):
        self.name = _name
        self.exts = { PC: '-PCs.nc', EOF: '-EOFs.nc' }
        self.directory = _dir if _dir is not None else os.path.join( os.path.expanduser( "~/results" ), _name )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    @staticmethod
    def new( baseDir, name ):
        if baseDir is None: baseDir = os.path.expanduser( "~/results" )
        directory = os.path.join( baseDir, name )
        return Project( directory, name )

    def outfilePath(self, experiment, rType ):
        # type: (str, int) -> str
        return os.path.join( self.directory, experiment + self.exts.get( rType, "Invalid response type: " + str(rType) ) )

class Experiment(object):

    def __init__(self, _project, _start_year, _end_year, _nModes, _variable ):
        self.exts = { PC: '-PCs.nc', EOF: '-EOFs.nc' }
        self.project = _project
        self.start_year = _start_year
        self.end_year = _end_year
        self.nModes = _nModes
        self.variable = _variable
        self.id = self.project.name + '_' + str(self.start_year) + '-' + str(self.end_year) + '_M' + str(self.nModes) + "_" + self.variable.id

    def serialize( self, lines ):
        lines.append( "@E:" + ",".join( [ "None", self.project.name, str(self.start_year), str(self.end_year), str(self.nModes), self.variable.varname, str(self.variable.level) ] ) )

    @staticmethod
    def deserialize( spec ):
        # type: (str) -> Experiment
        toks = spec.split(":")[1].split(",")
        return Experiment( Project( None, toks[1] ),  int(toks[2]), int(toks[3]), int(toks[4]), Variable( toks[5], toks[6] ) )

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

