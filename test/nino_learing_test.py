from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.batch import EpocServer
from cliMLe.trainingData import *
from time import time
from datetime import datetime, date, time
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
outDir = os.path.expanduser("~/results")

projectName = "MERRA2_EOFs"
start_year = 1980
end_year = 2015
nModes = 32
variables = [ Variable("ts"), Variable("zg",80000), Variable("zg",50000) ]
project = Project(outDir,projectName)
pcDataset = PCDataset( [ Experiment(project,start_year,end_year,nModes,variable) for variable in variables ] )

batchSize = 100
nEpocs = 100
validation_fraction = 0.1
lag = 0
nHiddenUnits = 32
timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
time_range = ( "1980-{0}-1".format(lag+1), "2014-12-1" if lag == 0 else "2015-{0}-1".format(lag) )
logDir = os.path.expanduser("~/results/logs/{}".format( projectName + "_" + timestamp ) )
plotPrediction = False

print "Saving results to log dir: " + logDir

td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "amo_timeseries_mon" ], time_range ) # , "pdo_timeseries_mon", "indian_ocean_dipole", "nino34"
dset = TrainingDataset( [td] )

model = Sequential()
model.add( Dense(units=nHiddenUnits, activation='relu', input_dim=pcDataset.getInputDimension() ) )
model.add( Dense( units=dset.output_size ) )
model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )

x = pcDataset.getEpoch()
y = dset.getEpoch()
tensorboard = TensorBoard( log_dir=logDir, histogram_freq=0, write_graph=True )
model.fit( x, y, batch_size=batchSize, epochs=nEpocs, validation_split=validation_fraction, shuffle=True, callbacks=[tensorboard] )

if plotPrediction:
    prediction = model.predict(x)
    plt.title("Training data with Prediction ({0}->nino34, lag {1}) {2}-{3} ({4} Epochs)".format(pcDataset.getVariableIds(),lag,start_year,end_year,nEpocs))
    plt.plot(prediction, label = "prediction")
    plt.plot(y, label = "training data")
    plt.legend()
    plt.show()