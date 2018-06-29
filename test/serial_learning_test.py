from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
import multiprocessing as mp
import time, keras
from datetime import datetime
outDir = os.path.expanduser("~/results")

projectName = "MERRA2_EOFs"
start_year = 1980
end_year = 2015
nModes = 32
variables = [ Variable("ts") ]
project = Project(outDir,projectName)
pcDataset = PCDataset( [ Experiment(project,start_year,end_year,nModes,variable) for variable in variables ] )

prediction_lag = 0
training_time_range = ( "1980-{0}-1".format(prediction_lag+1), "2014-12-1" if prediction_lag == 0 else "2015-{0}-1".format(prediction_lag) )
td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "amo_timeseries_mon" ], training_time_range ) # , "pdo_timeseries_mon", "indian_ocean_dipole", "nino34"
trainingDataset = TrainingDataset( [td] )

nIterations = 50
batchSize = 100
nEpocs = 300
validation_fraction = 0.1
hiddenLayers = [8]
activation = "relu"
plotPrediction = True

def learning_model_factory():
    print "Creating Learning Model"
    return LearningModel( pcDataset, trainingDataset, batch=batchSize, epocs=nEpocs, vf=validation_fraction, hidden=hiddenLayers, activation=activation )

result = LearningModel.serial_execute( learning_model_factory, nIterations )
print "Got Best result, val_loss = " + str( result.val_loss )

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->nino34, lag {1}) {2}-{3} ({4} Epochs)".format(pcDataset.getVariableIds(),prediction_lag,start_year,end_year,nEpocs)
    learningModel = learning_model_factory( )
    learningModel.plotPrediction( result, plot_title )