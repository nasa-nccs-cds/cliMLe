from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
from cliMLe.learning import FitResult
import time, keras
from datetime import datetime
from keras.callbacks import TensorBoard, History
import tensorflow as tf
import matplotlib.pyplot as plt
outDir = os.path.expanduser("~/results")

projectName = "MERRA2_EOFs"
start_year = 1980
end_year = 2015
nModes = 32
variables = [ Variable("ts") ]
project = Project(outDir,projectName)
pcDataset = PCDataset( [ Experiment(project,start_year,end_year,nModes,variable) for variable in variables ] )
nInterations = 100

batchSize = 50
nEpocs = 300
validation_fraction = 0.1
lag = 0
nHiddenUnits = 16
timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
time_range = ( "1980-{0}-1".format(lag+1), "2014-12-1" if lag == 0 else "2015-{0}-1".format(lag) )
logDir = os.path.expanduser("~/results/logs/{}".format( projectName + "_" + timestamp ) )
plotPrediction = True

td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "amo_timeseries_mon" ], time_range ) # , "pdo_timeseries_mon", "indian_ocean_dipole", "nino34"
dset = TrainingDataset( [td] )
x = pcDataset.getEpoch()
y = dset.getEpoch()
tensorboard = TensorBoard( log_dir=logDir, histogram_freq=0, write_graph=True )

bestFitResult = None  # type: FitResult
for iterIndex in range(nInterations):
    model = Sequential()
    model.add( Dense(units=nHiddenUnits, activation='relu', input_dim=pcDataset.getInputDimension() ) )
    model.add( Dense( units=dset.output_size ) )
    model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )
    history = model.fit( x, y, batch_size=batchSize, epochs=nEpocs, validation_split=validation_fraction, shuffle=True, callbacks=[tensorboard], verbose=0 )  # type: History
    val_loss = history.history['val_loss']
    current_min_loss_val, current_min_loss_index = min( (val_loss[i],i) for i in xrange(len(val_loss)) )
    if not bestFitResult or (current_min_loss_val < bestFitResult.val_loss):
        bestFitResult = FitResult.new( history, current_min_loss_val, current_min_loss_index )
    print "Iteration {0}, val_loss = {1}, val_loss index = {2}, min val_loss = {3}".format( iterIndex, current_min_loss_val, current_min_loss_index, bestFitResult.val_loss )

if plotPrediction:
    prediction = bestFitResult.model().predict(x)
    plt.title("Training data with Prediction ({0}->nino34, lag {1}) {2}-{3} ({4} Epochs)".format(pcDataset.getVariableIds(),lag,start_year,end_year,nEpocs))
    plt.plot(prediction, label = "prediction")
    plt.plot(y, label = "training data")
    plt.legend()
    plt.show()