from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, BATCH
from cliMLe.batch import EpocServer
from cliMLe.trainingData import *
from time import time
from datetime import datetime, date, time
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt

projectName = "MERRA2_EOFs"
varName = "zg"
outDir = os.path.expanduser("~/results")
start_year = 1980
end_year = 2015
nModes = 32
batchSize = 100
nEpocs = 1000
nHiddenUnits = 16
timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + varName
time_range_lag0 = ( "1980-1-1", "2014-12-1" )
time_range_lag1 = ( "1980-2-1", "2015-1-1" )
time_range_lag2 = ( "1980-3-1", "2015-2-1" )
time_range_lag6 = ( "1980-7-1", "2015-6-1" )
logDir = os.path.expanduser("~/results/logs/{}".format(experiment + "-lag6"  ))
print "Saving results to log dir: " + logDir

td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], time_range_lag6 )
dset = TrainingDataset( [td] )

project = Project(outDir,projectName)
eserv = EpocServer( project, experiment, dset, 0.1 )

model = Sequential()
model.add( Dense(units=nHiddenUnits, activation='relu', input_dim=nModes ) )
model.add( Dense( units=eserv.output_size ) )
model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )

x_train, y_train = eserv.getTrain()
X_test, Y_test = eserv.getValidation()
x, y = eserv.getEpoch()

tensorboard = TensorBoard( log_dir=logDir, histogram_freq=0, write_graph=True )

model.fit( x_train, y_train, batch_size=batchSize, epochs=nEpocs, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tensorboard] )

prediction = model.predict(x)
plt.title("Training data with Prediction (nino34) 1980-2015 (1000 Epochs)")
plt.plot(prediction, label = "prediction", color = "r")
plt.plot(y, label = "training data", color = "b")
plt.legend()
plt.show()