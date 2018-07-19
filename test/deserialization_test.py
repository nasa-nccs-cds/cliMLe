from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
from cliMLe.dataProcessing import CTimeRange, CDuration
from cliMLe.inputData import CDMSInputSource, InputDataset
outDir = os.path.expanduser("~/results")

instance = "20CRv2c_EOFs_2018-07-19_16:06:14.560105"

( learningModel, result ) = LearningModel.loadInstance( instance )
varIds = learningModel.inputs.getVariableIds()
trainingId = learningModel.outputs.id()
plot_title = "Training data with Prediction ({0}->{1}) (loss: {2}, Epochs: {3})".format(varIds,trainingId,result.val_loss,result.nEpocs)
learningModel.plotPrediction( result, "Monsoon Prediction with {0}".format(trainingId) )
learningModel.plotPerformance( result, plot_title )
