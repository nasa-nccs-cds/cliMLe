from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
outDir = os.path.expanduser("~/results")

pname = "20CRv2c"
projectName = pname + "_EOFs"
start_year = 1851
end_year = 2012
nModes = 64
nTS = 2
smooth = 0

variables = [ Variable("ts"), Variable( "zg", 80000 ), Variable( "zg", 50000 ), Variable( "zg", 25000 ) ]
project = Project(outDir,projectName)
pcDataset = PCDataset( [ Experiment(project,start_year,end_year,nModes,variable) for variable in variables ], nts = nTS, smooth = smooth )

prediction_lag = 1
nInterationsPerProc = 10
batchSize = 100
nEpocs = 200
validation_fraction = 0.15
hiddenLayers = [100]
activation = "relu"
plotPrediction = True

training_time_range = ( "1980-{0}-1".format(prediction_lag+1), "2014-12-1" if prediction_lag == 0 else "2015-{0}-1".format(prediction_lag) )
td = IITMDataSource( "AI", "monthly", training_time_range )
trainingDataset = TrainingDataset( [td], pcDataset )

#ref_time_range = ( "1980-1-1", "2014-12-1" )
#ref_ts = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], ref_time_range )

def learning_model_factory( weights = None ):
    return LearningModel( pcDataset, trainingDataset, batch=batchSize, epocs=nEpocs, vf=validation_fraction, hidden=hiddenLayers, activation=activation, weights=weights )

result = LearningModel.parallel_execute( learning_model_factory, nInterationsPerProc )
print "Got Best result, valuation loss = " + str( result.val_loss ) + " training loss = " + str( result.train_loss )

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->IITM-AI, lag {1}) {2}-{3} (loss: {4}, Epochs: {5})".format(pcDataset.getVariableIds(),prediction_lag,start_year,end_year,result.val_loss,result.nEpocs)
    learningModel = learning_model_factory()
    learningModel.plotPrediction( result, plot_title )