from cliMLe.learning import FitResult, LearningModel
from cliMLe.svd import Eof
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
            c.append( lmod.outputData[iPoint] )
        return np.array( x ), np.array( y ), np.array( c )

    @classmethod
    def plotModeDistribution( cls, lmod ):
        # type: (LearningModel) -> None
        pcd = cls.getPCDistribution( lmod )



if __name__ == "__main__":
    instance = "ams-result-1"
    learningModel, result = LearningModel.loadInstance(instance)
    model = learningModel.getFittedModel(result)

    x, y, c = Visualizer.getPCDistribution( learningModel)

    print str( x )