from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
from cliMLe.dataProcessing import CTimeRange, CDuration
from cliMLe.inputData import CDMSInputSource, InputDataset
outDir = os.path.expanduser("~/results")

pname = "20CRv2c"
projectName = pname + "_EOFs"
nModes = 16
start_year = 1851
end_year = 2012
nTS = 1
smooth = 0
freq="Y"   # Yearly input/outputs
filter="ja"   # Filter months out of each year.
learning_range = CTimeRange.new( "1851-1-1", "2005-12-1" )

variables = [ Variable("ts"), Variable( "zg", 25000 ) ]  # [ Variable("ts"), Variable( "zg", 80000 ), Variable( "zg", 50000 ), Variable( "zg", 25000 ) ]
project = Project.new(outDir,projectName)
pcDataset = PCDataset( projectName, [ Experiment(project,start_year,end_year,64,variable) for variable in variables ], nts = nTS, smooth = smooth, filter=filter, nmodes=nModes, freq=freq, timeRange = learning_range )
inputDataset = InputDataset( [ pcDataset ] )

prediction_lag = CDuration.years(1)
nInterationsPerProc = 10
batchSize = 150
maxTrainingLoss = 0.4
nEpocs = 300
learnRate = 0.005
momentum=0.00
decay=0.0
nesterov=False
validation_fraction = 0.2
hiddenLayers = [ pcDataset.getInputDimension() ]
activation = "relu"
plotPrediction = True
orthoWts=False

tds = [ IITMDataSource( domain, "JJAS" ) for domain in [ "AI" ] ] # [ "AI", "EPI", "NCI", "NEI", "NMI", "NWI", "SPI", "WPI"]
trainingDataset = TrainingDataset.new( tds, pcDataset, prediction_lag )

#ref_time_range = ( "1980-1-1", "2014-12-1" )
#ref_ts = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], ref_time_range )

def learning_model_factory( weights = None ):
    return LearningModel( inputDataset, trainingDataset, batch=batchSize, lrate=learnRate, momentum=momentum, decay=decay, nesterov=nesterov, orthoWts=orthoWts, epocs=nEpocs, vf=validation_fraction, hidden=hiddenLayers, max_loss=maxTrainingLoss, activation=activation, weights=weights )

result = LearningModel.parallel_execute( learning_model_factory, nInterationsPerProc )
print "Got Best result, valuation loss = " + str( result.val_loss ) + " training loss = " + str( result.train_loss )

learningModel = learning_model_factory()
learningModel.serialize( inputDataset, trainingDataset, result )

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->IITM, lag {1}) {2}-{3} (loss: {4}, Epochs: {5})".format(pcDataset.getVariableIds(),prediction_lag,start_year,end_year,result.val_loss,result.nEpocs)
    learningModel.plotPrediction( result, "Monsoon Prediction with IITM" )
    learningModel.plotPerformance( result, plot_title )

