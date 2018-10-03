from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel

instance = "good_1"

( learningModel, result ) = LearningModel.loadInstance( instance )
varIds = learningModel.inputs.getVariableIds()
trainingId = learningModel.outputs.id()
plot_title = "Training data with Prediction ({0}->{1}) (loss: {2}, Epochs: {3})".format(varIds,trainingId,result.val_loss,result.nEpocs)
learningModel.plotPrediction( result, "All-India Monsoon Rainfall Prediction, 1 Year Lag Time" )
