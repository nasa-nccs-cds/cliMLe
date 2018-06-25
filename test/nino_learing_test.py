from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, BATCH
from cliMLe.batch import BatchServer
from cliMLe.trainingData import *
from time import time
from keras.callbacks import TensorBoard

projectName = "MERRA2_EOFs"
varName = "ts"
outDir = os.path.expanduser("~/EOFs")
start_year = 1980
end_year = 2015
nModes = 32
batchSize = 80
nEpocs = 500
experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + varName
time_range_lag0 = ( "1980-1-1", "2014-12-1" )
time_range_lag1 = ( "1980-2-1", "2015-1-1" )
time_range_lag2 = ( "1980-3-1", "2015-2-1" )
logDir = os.path.expanduser("~/EOFs/logs/{}".format(time()))

td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "nino34" ], time_range_lag0 )
dset = TrainingDataset( [td] )

project = Project(outDir,projectName)
bserv = BatchServer( project, experiment, dset, batchSize, BATCH )

model = Sequential()
model.add( Dense(units=16, activation='relu', input_dim=nModes ) )
model.add( Dense( units=bserv.output_size ) )
model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )
tensorboard = TensorBoard( log_dir=logDir )

for iEpoch in range( nEpocs ):
    for iBatch in range( bserv.numBatches ):
        x_train, y_train = bserv.getNextBatch()
        model.train_on_batch( x_train, y_train )
    x_test, y_test = bserv.getCurrentBatch()
    loss_and_metrics = model.evaluate( x_test, y_test, batch_size=batchSize )
    print "Epoc {0}, loss_and_metrics= {1}".format( str(iEpoch), str(loss_and_metrics) )

