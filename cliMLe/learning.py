from keras.models import Sequential
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
import time, keras
from datetime import datetime
from keras.callbacks import TensorBoard, History
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging
logging.basicConfig(filename='/tmp/cliMLe.log',level=logging.DEBUG)
outDir = os.path.expanduser("~/results")

class FitResult:

    def __init__( self, _history, _val_loss, _nEpocs ):
        self.history = _history
        self.val_loss = _val_loss
        self.nEpocs = _nEpocs

    def valLossSeries(self):
        return self.history.history['val_loss']

    def model(self):
        return self.history.model

    def __lt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss < other.val_loss

    def __gt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss > other.val_loss

    def __eq__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss == other.val_loss

    def getBest( self, results ):
        # type: (list[FitResult]) -> FitResult
        bestResult = None
        for result in results:
            if bestResult is None or result < bestResult:
                bestResult = result
        return bestResult

    def plotPrediction( self, inputData, outputData, title ):
        prediction = self.model().predict( inputData )
        plt.title( title )
        plt.plot( prediction, label = "prediction")
        plt.plot( outputData, label = "training data" )
        plt.legend()
        plt.show()

class LearningModel:

    def __init__( self, inputDataset, trainingDataset,  **kwargs ):
        self.inputs = inputDataset                  # type: PCDataset
        self.outputs = trainingDataset              # type: TrainingDataset
        self.batchSize = kwargs.get( 'batch', 50 )
        self.nEpocs = kwargs.get( 'epocs', 300 )
        self.validation_fraction = kwargs.get(  'vf', 0.1 )
        self.hidden = kwargs.get(  'hidden', [16] )
        self.activation = kwargs.get(  'activation', 'relu' )
        self.timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
        self.projectName = self.inputs.experiments[0].project.name
        self.logDir = os.path.expanduser("~/results/logs/{}".format( self.projectName + "_" + self.timestamp ) )
        self.inputData = self.inputs.getEpoch()
        self.outputData = self.outputs.getEpoch()
        self.tensorboard = TensorBoard( log_dir=self.logDir, histogram_freq=0, write_graph=True )
        self.bestFitResult = None # type: FitResult

    def execute( self, nIterations=1, procIndex=0 ):
        # type: () -> FitResult
        try:
            for iterIndex in range(nIterations):
                model = Sequential()
                for hIndex in range(len(self.hidden)):
                    if hIndex == 0:     model.add( Dense(units=self.hidden[hIndex], activation=self.activation, input_dim=self.inputs.getInputDimension() ) )
                    else:               model.add( Dense(units=self.hidden[hIndex], activation=self.activation ) )
                model.add( Dense( units=self.outputs.output_size ) )
                model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )
                history = model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=True, callbacks=[self.tensorboard], verbose=0 )  # type: History
                self.updateHistory( history, iterIndex, procIndex=0 )
        except Exception as err:
            logging.error( "PROC[{0}]: {1}".format( procIndex, str(err) ) )
        return self.bestFitResult

    def updateHistory( self, history, iterIndex, procIndex ):
        # type: (History, int) -> History
        val_loss = history.history['val_loss']
        current_min_loss_val, current_min_loss_index = min( (val_loss[i],i) for i in xrange(len(val_loss)) )
        if not self.bestFitResult or (current_min_loss_val < self.bestFitResult.val_loss ):
            self.bestFitResult = FitResult( history, current_min_loss_val, current_min_loss_index )
        logging.info( "PROC[{0}]: Iteration {1}, val_loss = {2}, val_loss index = {3}, min val_loss = {4}".format( procIndex, iterIndex, current_min_loss_val, current_min_loss_index, self.bestFitResult.val_loss ) )
        return history

    def plotPrediction( self, title, result = None ):
        plotResult = self.bestFitResult if result is None else result
        plotResult.plotPrediction( self.inputData,self.outputData, title  )

    @classmethod
    def execute_learning_model( cls, procIndex, nInterations, resultQueue, lmodel_factory):
        try:
            learningModel = lmodel_factory()
            result = learningModel.execute( nInterations, procIndex )
            resultQueue.put( result )
        except Exception as err:
            logging.error( "PROC[{0}]: {1}".format( procIndex, str(err) ) )

    @classmethod
    def parallel_execute(cls, learning_model_factory, nIterPerProc, nProc=mp.cpu_count() ):
        # type: (object, int, int) -> FitResult
        resultQueue = mp.Queue()
        learning_processes = []

        for iProc in range(nProc):
            print " ** Executing process " + str(iProc)
            proc = mp.Process( target=cls.execute_learning_model, args=( iProc, nIterPerProc, resultQueue, learning_model_factory ) )
            learning_processes.append(proc)

        results = []
        for iProc in range(mp.cpu_count()):
            print " ** Waiting for result from process " + str(iProc)
            results.append( resultQueue.get() )

        return FitResult.getBest( results )

