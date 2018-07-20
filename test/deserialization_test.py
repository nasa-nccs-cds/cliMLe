from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel

instance = "20CRv2c_EOFs_2018-07-20_14:07:54.998592"

( learningModel, result ) = LearningModel.loadInstance( instance )
varIds = learningModel.inputs.getVariableIds()
trainingId = learningModel.outputs.id()
plot_title = "Training data with Prediction ({0}->{1}) (loss: {2}, Epochs: {3})".format(varIds,trainingId,result.val_loss,result.nEpocs)
learningModel.plotPrediction( result, "Monsoon Prediction with {0}".format(trainingId) )
learningModel.plotPerformance( result, plot_title )
