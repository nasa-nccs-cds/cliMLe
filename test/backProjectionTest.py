from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
instance = "good_1"

result = LearningModel.getBackProjection( instance )

print str( result.shape )