from cliMLe.climatele import Project, Variable, Experiment, ClimateleDataset
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
nTS = 1
smooth = 0
learning_range = CTimeRange.new( "1980-1-1", "2014-12-30" )

prediction_lag = CDuration.months(6)
nInterations = 10
batchSize = 100
nEpocs = 200
validation_fraction = 0.15
hiddenLayers = [100]
activation = "relu"
plotPrediction = True

variables = [ Variable("ts") ] # , Variable( "zg", 50000 ) ]
project = Project(outDir,projectName)
pcDataset = ClimateleDataset([Experiment(project, start_year, end_year, nModes, variable) for variable in variables], nts = nTS, smooth = smooth, timeRange = learning_range)
td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ] ) # , "pdo_timeseries_mon", "indian_ocean_dipole", "nino34"
trainingDataset = TrainingDataset.new( [td], pcDataset, prediction_lag )

def learning_model_factory():
    return LearningModel( pcDataset, trainingDataset, batch=batchSize, epocs=nEpocs, vf=validation_fraction, hidden=hiddenLayers, activation=activation )

result = LearningModel.serial_execute( learning_model_factory, nInterations )
print "Got Best result, val_loss = " + str( result.val_loss )

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->nino34, lag {1}) {2}-{3} ({4} Epochs)".format(pcDataset.getVariableIds(),prediction_lag,start_year,end_year,nEpocs)
    learningModel = learning_model_factory( )
    learningModel.plotPrediction( result, plot_title )