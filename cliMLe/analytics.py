from cliMLe.learning import LearningModel
from cliMLe.climatele import Experiment, ClimateleDataset
from cliMLe.project import PC, EOF
from cliMLe.svd import Eof
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:

    @classmethod
    def getModes( cls, lmod, nModes ):
        # type: (LearningModel,int) -> np.array
        solver = Eof( lmod.inputData )
        modes = solver.eofs(neofs=nModes)
        return modes

    @classmethod
    def getPCDistribution( cls, lmod ):
        # type: (LearningModel) -> ( np.array, np.array, np.array )
        modes = cls.getModes(lmod,2)
        x, y, c = [], [], []
        for iPoint in range( modes.shape[1] ):
            input = lmod.inputData[iPoint]
            x.append( np.dot( modes[0], input ) )
            y.append( np.dot(modes[1], input) )
            c.append( lmod.outputData[iPoint][0] )
        return np.array( x ), np.array( y ), np.array( c )

    @classmethod
    def plotModeDistribution( cls, lmod ):
        # type: (LearningModel) -> None
        x, y, c = Visualizer.getPCDistribution(lmod)

        fig, ax = plt.subplots()
        cax = plt.scatter( x, y, None, c, None, "jet" )
        cbar = fig.colorbar( cax )
        plt.title( ' PC Distribution Plot ' )
        plt.show()

    @classmethod
    def getWeightedPatterns( cls, lmod ):
        # type: (LearningModel) -> List[CdmsFile]
        cdset = None # type: ClimateleDataset
        dsets = []
        for cdset in lmod.inputs.sources:
            for exp in cdset.experiments:
                dsets.append( exp.getDataset( EOF ) )
        return dsets


if __name__ == "__main__":
    instance = "ams-result-1"
    learningModel, result = LearningModel.loadInstance(instance)
#    model = learningModel.getFittedModel(result)
    dsets = Visualizer.getWeightedPatterns( learningModel )

