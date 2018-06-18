from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import BatchServer, Project, BATCH

projectName = "MERRA2_EOFs"
varName = "ts"
data_path = 'https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/Reanalysis/NASA-GMAO/GEOS-5/MERRA2/mon/atmos/' + varName + '.ncml'
outDir = "/tmp/"
start_year = 1980
end_year = 2015
nModes = 32
batchSize = 100
nEpocs = 10

experiment = projectName + '_'+str(start_year)+'-'+str(end_year) + '_M' + str(nModes) + "_" + varName


project = Project(outDir,projectName)
bserv = BatchServer( project, experiment, batchSize, BATCH )

model = Sequential()
model.add( Dense(units=16, activation='relu', input_dim=nModes ) )
model.add( Dense( units=8 ) )
model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )

y_train = []      # Add training data.

for iEpoch in range( nEpocs ):
    for iBatch in range( bserv.batchSize ):
        x_train = bserv.getNextBatch()
        model.train_on_batch( x_train, y_train )

# loss_and_metrics = model.evaluate( x_test, y_test, batch_size=batchSize )