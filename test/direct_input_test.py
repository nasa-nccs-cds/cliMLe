from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
from cliMLe.dataProcessing import CTimeRange, CDuration
from cliMLe.inputData import CDMSInputSource, InputDataset
from cliMLe.layers import Layer, Layers
import cdms2 as cdms
import cdtime, math, cdutil, time, os
outDir = os.path.expanduser("~/results")

plotPrediction = True
plotVerification = False

trainStartDate="1851-1-1"
trainEndDate="2005-12-1"
verificationSplitDate= "1861-1-1"

pname = "20CRv2c"
projectName = pname + "_EOFs"
nModes = 32
start_year = None
end_year = None
nTS = 1
smooth = 0
freq="Y"   # Yearly input/outputs
filter="8"   # Filter months out of each year.
variables = [ Variable("ts"), Variable( "zg", 80000 ) ]  # [ Variable("ts"), Variable( "zg", 80000 ), Variable( "zg", 50000 ), Variable( "zg", 25000 ) ]

if pname.lower() == "merra2":
    start_year = 1980
    end_year = 2015
elif pname.lower() == "20crv2c":
    start_year = 1851
    end_year = 2012

input_vars = {}

for var in variables:
    if pname.lower() == "merra2": data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/Reanalysis/NASA-GMAO/GEOS-5/MERRA2/mon/atmos/' + var.varname + '.ncml'
    elif pname.lower() == "20crv2c": data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/reanalysis/20CRv2c/mon/atmos/' + var.varname + '.ncml'
    else: raise Exception( "Unknown project name: " + pname )
    start_time = cdtime.comptime(start_year)
    end_time = cdtime.comptime(end_year)

    #------------------------------ READ DATA ------------------------------

    read_start = time.time()
    print "Computing Eofs for variable " + var.varname + ", level = " + str(var.level) + ", nModes = " + str(nModes)

    if var.level:
        experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + var.varname + "-" + str(var.level)
        f = cdms.open(data_path)
        variable = f( var.varname, latitude=(-80,80), level=(var.level,var.level), time=(start_time,end_time) )  # type: cdms.tvariable.TransientVariable
    else:
        experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + var.varname
        f = cdms.open(data_path)
        variable = f( var.varname, latitude=(-80,80), time=(start_time,end_time) )  # type: cdms.tvariable.TransientVariable

    input_vars[experiment] = variable

    print "Completed data read in " + str(time.time()-read_start) + " sec "


prediction_lag = CDuration.years(1)
nInterationsPerProc = 20
batchSize = 200
nEpocs = 1000
learnRate = 0.005
momentum=0.9
decay=0.002
loss_function="mse"
nesterov=False
validation_fraction = 0.2
stopCondition="minValTrain"
earlyTermIndex=50
shuffle=True
nHiddenUnits = 64

initWtsMethod="lecun_normal"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
orthoWts=False

tds = [ IITMDataSource( domain, "JJAS" ) for domain in [ "AI" ] ] # [ "AI", "EPI", "NCI", "NEI", "NMI", "NWI", "SPI", "WPI"]
trainingDataset = TrainingDataset.new( tds, pcDataset, prediction_lag )

#ref_time_range = ( "1980-1-1", "2014-12-1" )
#ref_ts = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], ref_time_range )

layers = [ Layer( "dense", nHiddenUnits, activation = "relu", kernel_initializer = initWtsMethod ), Layer( "dense", trainingDataset.getOutputSize(), kernel_initializer = initWtsMethod ) ]

def learning_model_factory( inputs=inputDataset, target=trainingDataset, weights=None ):
    return LearningModel( inputs, target, layers, batch=batchSize, earlyTermIndex=earlyTermIndex, lrate=learnRate, stop_condition=stopCondition, shuffle=shuffle, loss_function=loss_function, momentum=momentum, decay=decay, nesterov=nesterov, orthoWts=orthoWts, epocs=nEpocs, vf=validation_fraction, weights=weights )

result = LearningModel.parallel_execute( learning_model_factory, nInterationsPerProc )

learningModel = learning_model_factory()
learningModel.serialize( inputDataset, trainingDataset, result )

if plotPrediction:
    plot_title = "Training data with Prediction ({0}->IITM, lag {1}) {2}-{3} (loss: {4}, Epochs: {5})".format(pcDataset.getVariableIds(),prediction_lag,proj_start_year,proj_end_year,result.val_loss,result.nEpocs)
    learningModel.plotPrediction( result, "Monsoon Prediction with IITM" )
    learningModel.plotPerformance( result, plot_title )

if plotVerification:
    verification_range = CTimeRange.new(trainStartDate, verificationSplitDate)
    pcVerificationDataset = PCDataset( projectName, experiments, nts = nTS, smooth = smooth, filter=filter, nmodes=nModes, freq=freq, timeRange = verification_range )
    verInputDataset = InputDataset( [ pcVerificationDataset ] )
    verTargetDataset = TrainingDataset.new( tds, pcVerificationDataset, prediction_lag )
    verificationModel = learning_model_factory( verInputDataset, verTargetDataset )
    verificationModel.plotPrediction( result, "Verification" )
