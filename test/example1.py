from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, BATCH
from cliMLe.batch import BatchServer
from cliMLe.trainingData import *

projectName = "MERRA2_EOFs"
varName = "zg"
outDir = os.path.expanduser("~/results/")
start_year = 1980
end_year = 2015
nModes = 20
batchSize = 50
nEpocs = 100
experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + varName
time_range_lag0 = ( "1980-1-1", "2014-12-1" )
time_range_lag1 = ( "1980-2-1", "2015-1-1" )
time_range_lag2 = ( "1980-3-1", "2015-2-1" )

td = ProjectDataSource( "HadISST_1.cvdp_data.1980-2017", [ "indian_ocean_dipole", "pdo_timeseries_mon", "amo_timeseries_mon" ], time_range_lag1 )
dset = TrainingDataset( [td] )

project = Project(outDir,projectName)
bserv = BatchServer( project, experiment, dset, batchSize, BATCH )

model = Sequential()
model.add( Dense(units=16, activation='relu', input_dim=nModes ) )
model.add( Dense( units=bserv.output_size ) )
model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )

for iEpoch in range( nEpocs ):
    for iBatch in range( bserv.numBatches ):
        x_train, y_train = bserv.getNextBatch()
        print "Training on Epoc {0}, Batch {1}, input shape = {2}, output shape = {3} ".format( str(iEpoch), str(iBatch), str(x_train.shape), str(y_train.shape))
        model.train_on_batch( x_train, y_train )

    x_test, y_test = bserv.getCurrentBatch()
    loss_and_metrics = model.evaluate( x_test, y_test, batch_size=batchSize )
    print str( loss_and_metrics )
