from __future__ import absolute_import, division, print_function
from cliMLe.project import Project, Variable, Experiment, ODAPVariable
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
from cliMLe.dataProcessing import CTimeRange, CDuration
from cliMLe.inputData import ReanalysisInputSource, InputDataset
from cliMLe.climatele import ClimateleDataset
from cliMLe.layers import Layer, Layers
import tensorflow as tf
import cdms2 as cdms
import cdtime, math, cdutil, time, os
outDir = os.path.expanduser("~/results")

plotPrediction = True

pname = "20CRv2c"
trainStartDate="1851-1-1"
trainEndDate="2005-12-1"
learning_range = CTimeRange.new(trainStartDate, trainEndDate)

def getDsetUrl(pname, varname):
    if pname.lower() == "merra2":
        data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/Reanalysis/NASA-GMAO/GEOS-5/MERRA2/mon/atmos/' + varname + '.ncml'
    elif pname.lower() == "20crv2c":
        data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/reanalysis/20CRv2c/mon/atmos/' + varname + '.ncml'
    else:
        raise Exception(" Unrecognized project: " + pname)
    return data_path

projectName = pname + "_EOFs"
variables = [ODAPVariable("ts", getDsetUrl(pname, "ts"))]
proj_start_year = 1851
proj_end_year = 2012
filter = "8"
freq = "Y"
nTS = 1
nModes = 32
smooth = 0

inputs = ReanalysisInputSource(projectName, variables, latitude=(-80, 80), filter=filter, freq=freq, time = learning_range )
reanalysisInputDataset = InputDataset([inputs])
weights = reanalysisInputDataset.getEOFWeights(nModes)
biases = np.zeros( [nModes] )
print ("Weights Shape = " + str( weights.shape ))
print ("Weights Sample = " + str( weights[0] ))

variables1 = [ Variable("ts"), Variable( "zg", 80000 ) ]  # [ Variable("ts"), Variable( "zg", 80000 ), Variable( "zg", 50000 ), Variable( "zg", 25000 ) ]
project = Project.new(outDir,projectName)
experiments = [ Experiment(project,proj_start_year,proj_end_year,64,variable) for variable in variables1 ]
pcDataset = ClimateleDataset(projectName, experiments, nts = nTS, smooth = smooth, filter=filter, nmodes=nModes, freq=freq, timeRange = learning_range)
pcInputDataset = InputDataset( [ pcDataset ] )

prediction_lag = CDuration.years(1)
nInterationsPerProc = 5
nShuffles = 3
batchSize = 200
nEpocs = 500
learnRate = 0.005
momentum=0.9
decay=0.002
loss_function="mse"
nesterov=False
validation_fraction = 0.2
stopCondition="minValTrain"
earlyTermIndex=20
shuffle=True
nHiddenUnits = 64
initWtsMethod="lecun_normal"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
orthoWts=False
parallelExe = True
eagerExe = False

input_epoch = inputs.getEpoch()
input_shape = input_epoch.shape
print ("Epoch Shape = " + str( input_shape ))

tds = [ IITMDataSource( domain, "JJAS" ) for domain in [ "AI" ] ] # [ "AI", "EPI", "NCI", "NEI", "NMI", "NWI", "SPI", "WPI"]
trainingDataset = TrainingDataset.new( tds, inputs, prediction_lag )

#ref_time_range = ( "1980-1-1", "2014-12-1" )
#ref_ts = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], ref_time_range )

layers3R = [ Layer( "dense", nModes, trainable=False, kernel_initializer = initWtsMethod ),
           Layer( "dense", nHiddenUnits, activation = "relu", kernel_initializer = initWtsMethod ),
           Layer( "dense", trainingDataset.getOutputSize(), kernel_initializer = initWtsMethod ) ]


layers3 = [ Layer( "dense", nModes, trainable=False, weights=[ weights, biases ] ),
           Layer( "dense", nHiddenUnits, activation = "relu", kernel_initializer = initWtsMethod ),
           Layer( "dense", trainingDataset.getOutputSize(), kernel_initializer = initWtsMethod ) ]

layers2 = [ Layer( "dense", nModes, activation = "relu", kernel_initializer = initWtsMethod ),
            Layer( "dense", trainingDataset.getOutputSize(), kernel_initializer = initWtsMethod ) ]

def direct_learning_model_factory(inputs=reanalysisInputDataset, target=trainingDataset, weights=None):
    return LearningModel( inputs, target, layers3, verbose=False, eager=eagerExe, batch=batchSize, earlyTermIndex=earlyTermIndex, lrate=learnRate, stop_condition=stopCondition, shuffle=shuffle, loss_function=loss_function, momentum=momentum, decay=decay, nesterov=nesterov, orthoWts=orthoWts, epocs=nEpocs, vf=validation_fraction, weights=weights )

def pc_learning_model_factory( inputs=pcInputDataset, target=trainingDataset, weights=None ):
    return LearningModel( inputs, target, layers2, verbose=False, eager=eagerExe, batch=batchSize, earlyTermIndex=earlyTermIndex, lrate=learnRate, stop_condition=stopCondition, shuffle=shuffle, loss_function=loss_function, momentum=momentum, decay=decay, nesterov=nesterov, orthoWts=orthoWts, epocs=nEpocs, vf=validation_fraction, weights=weights )

learning_model_factory = direct_learning_model_factory
result = LearningModel.fit( learning_model_factory, nInterationsPerProc, nShuffles, parallelExe )

def direct_learning_model_factory_trainable( ):
    layers3[0].setTrainable( True )
    return LearningModel( reanalysisInputDataset, trainingDataset, layers3, verbose=False, eager=eagerExe, batch=batchSize, earlyTermIndex=earlyTermIndex, lrate=learnRate, stop_condition=stopCondition, shuffle=shuffle, loss_function=loss_function, momentum=momentum, decay=decay, nesterov=nesterov, epocs=nEpocs, vf=validation_fraction, weights=result.final_weights )

learning_model_factory = direct_learning_model_factory_trainable
result = LearningModel.fit( learning_model_factory, nInterationsPerProc, nShuffles, parallelExe )

learningModel = learning_model_factory()

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->IITM, lag {1}) {2}-{3} (loss: {4}, Epochs: {5})".format(inputs.getVariableIds(),prediction_lag,trainStartDate,trainEndDate,result.val_loss,result.nEpocs)
    learningModel.plotPrediction( result, "Monsoon Prediction with IITM" )
    learningModel.plotPerformance( result, plot_title )





