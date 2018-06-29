from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.trainingData import *
import time, keras
from datetime import datetime
from keras.callbacks import TensorBoard, History
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging, traceback
outDir = os.path.join( os.path.expanduser("~"), "results" )
LOG_FILE = os.path.join( outDir, 'logs', 'cliMLe.log' )
try: os.makedirs( os.path.dirname(LOG_FILE) )
except: pass
logging.basicConfig(filename=LOG_FILE,level=logging.DEBUG)


class FitResult:

    @staticmethod
    def new( history, initial_weights, val_loss,  nEpocs):
        # type: (History, list[float], float, int) -> FitResult
        return FitResult( history.history['val_loss'], initial_weights, history.model.get_weights(),  val_loss, nEpocs )

    def __init__( self, _val_loss_history, _initial_weights, _final_weights,  _val_loss, _nEpocs ):
        self.val_loss_history = _val_loss_history
        self.initial_weights = _initial_weights
        self.final_weights = _final_weights
        self.val_loss = _val_loss
        self.nEpocs = _nEpocs

    def valLossHistory(self):
        return self.val_loss_history

    def __lt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss < other.val_loss

    def __gt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss > other.val_loss

    def __eq__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss == other.val_loss

    @staticmethod
    def getBest( results ):
        # type: (list[FitResult]) -> FitResult
        bestResult = None
        for result in results:
            if bestResult is None or result < bestResult:
                bestResult = result
        return bestResult

class LearningModel:

    def __init__( self, inputDataset, trainingDataset,  **kwargs ):
        self.inputs = inputDataset                  # type: PCDataset
        self.outputs = trainingDataset              # type: TrainingDataset
        self.batchSize = kwargs.get( 'batch', 50 )
        self.nEpocs = kwargs.get( 'epocs', 300 )
        self.validation_fraction = kwargs.get(  'vf', 0.1 )
        self.hidden = kwargs.get(  'hidden', [16] )
        self.weights = kwargs.get(  'weights', None )
        self.activation = kwargs.get(  'activation', 'relu' )
        self.timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
        self.projectName = self.inputs.experiments[0].project.name
        self.logDir = os.path.expanduser("~/results/logs/{}".format( self.projectName + "_" + self.timestamp ) )
        self.inputData = self.inputs.getEpoch()
        self.outputData = self.outputs.getEpoch()
        self.tensorboard = TensorBoard( log_dir=self.logDir, histogram_freq=0, write_graph=True )
        self.bestFitResult = None # type: FitResult

    def execute( self, nIterations=1, procIndex=-1 ):
        # type: () -> FitResult
        try:
            for iterIndex in range(nIterations):
                model = self.createSequentialModel()
                initial_weights = model.get_weights()
                history = model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=True, callbacks=[self.tensorboard], verbose=0 )  # type: History
                self.updateHistory( history, initial_weights, iterIndex, procIndex=0 )
        except Exception as err:
            logging.error( "PROC[{0}]: {1}".format( procIndex, str(err) ) )
            logging.error(traceback.format_exc())
        return self.bestFitResult

    def getInitialModel(self, fitResult):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()
        model.set_weights( fitResult.initial_weights )
        return model

    def getFittedModel( self, fitResult ):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()
        model.set_weights( fitResult.initial_weights )
        model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=fitResult.nEpocs, validation_split=self.validation_fraction, shuffle=True, verbose=0 )  # type: History
        return model

    def getFinalModel( self, fitResult ):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()
        model.set_weights( fitResult.final_weights )
        return model

    def getRerunModel( self, fitResult ):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()
        model.set_weights( fitResult.initial_weights )
        model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=True, verbose=0 )  # type: History
        return model

    def createSequentialModel( self ):
        # type: () -> Sequential
        model = Sequential()
        for hIndex in range(len(self.hidden)):
            if hIndex == 0:
                model.add(Dense(units=self.hidden[hIndex], activation=self.activation, input_dim=self.inputs.getInputDimension()))
            else:
                model.add(Dense(units=self.hidden[hIndex], activation=self.activation))
        model.add(Dense(units=self.outputs.output_size))
        model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
        if self.weights is not None: model.set_weights(self.weights)
        return model

    def updateHistory( self, history, initial_weights, iterIndex, procIndex ):
        # type: (History, int) -> History
        val_loss = history.history['val_loss']
        current_min_loss_val, current_min_loss_index = min( (val_loss[i],i) for i in xrange(len(val_loss)) )
        if not self.bestFitResult or (current_min_loss_val < self.bestFitResult.val_loss ):
            self.bestFitResult = FitResult.new( history, initial_weights, current_min_loss_val, current_min_loss_index )
        if procIndex < 0:   print "Executed iteration {0}, current loss = {1}, best loss = {2}".format(iterIndex, history.history[ 'val_loss'], self.bestFitResult.val_loss)
        else:               logging.info( "PROC[{0}]: Iteration {1}, val_loss = {2}, val_loss index = {3}, min val_loss = {4}".format( procIndex, iterIndex, current_min_loss_val, current_min_loss_index, self.bestFitResult.val_loss ) )
        return history

    def plotPrediction( self, fitResult, title ):
        plt.title(title)

        model1 = self.getFittedModel(fitResult)
        prediction1 = model1.predict( self.inputData )
        plt.plot(prediction1, label="prediction: fitted" )

        model2 = self.getFinalModel(fitResult)
        prediction2 = model2.predict( self.inputData )
        plt.plot(prediction2, label="prediction: final" )

        plt.plot( self.outputData, label="training data" )
        plt.legend()
        plt.show()

    @classmethod
    def serial_execute(cls, lmodel_factory, nInterations, procIndex=-1, resultQueue=None):
        # type: (object(), int, int, mp.Queue) -> FitResult
        try:
            learningModel = lmodel_factory()  # type: LearningModel
            result = learningModel.execute( nInterations, procIndex )
            if resultQueue is not None: resultQueue.put( result )
            return result
        except Exception as err:
            logging.error( "PROC[{0}]: {1}".format( procIndex, str(err) ) )
            logging.error(traceback.format_exc())

    @classmethod
    def terminate( cls, processes ):
        print " ** Terminating worker processes"
        for proc in processes:
            proc.terminate()

    @classmethod
    def parallel_execute(cls, learning_model_factory, nIterPerProc, nProc=mp.cpu_count() ):
        # type: ( object(), int, int) -> FitResult
        results = []
        learning_processes = []
        try:
            resultQueue = mp.Queue()

            print " ** Executing {0} iterations on each of {1} processors for a total of {2} interations, logging to {3}".format(nIterPerProc,nProc,nIterPerProc*nProc,LOG_FILE)

            for iProc in range(nProc):
                print " ** Executing process " + str(iProc)
                proc = mp.Process(target=cls.serial_execute, args=(learning_model_factory, nIterPerProc, iProc, resultQueue))
                proc.start()
                learning_processes.append(proc)

            for iProc in range(mp.cpu_count()):
                print " ** Waiting for result from process " + str(iProc)
                results.append( resultQueue.get() )

        except Exception:
            traceback.print_exc()
            cls.terminate( learning_processes )

        except KeyboardInterrupt:
            cls.terminate( learning_processes )

        return FitResult.getBest( results )

