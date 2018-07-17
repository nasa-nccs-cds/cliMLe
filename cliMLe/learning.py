from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from typing import Optional, Any

from cliMLe.pcProject import Project, Variable, Experiment, PCDataset
from cliMLe.inputData import InputDataset
from cliMLe.trainingData import *
import time, keras
from datetime import datetime
from keras.callbacks import TensorBoard, History
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import random, sys
import numpy as np
from keras.optimizers import SGD
import logging, traceback
outDir = os.path.join( os.path.expanduser("~"), "results" )
LOG_FILE = os.path.join( outDir, 'logs', 'cliMLe.log' )
try: os.makedirs( os.path.dirname(LOG_FILE) )
except: pass
logging.basicConfig(filename=LOG_FILE,level=logging.DEBUG)


class FitResult:

    @staticmethod
    def new( history, initial_weights, training_loss, val_loss,  nEpocs):
        # type: (History, list[np.ndarray], float, int) -> FitResult
        return FitResult( history.history['val_loss'], history.history['loss'], initial_weights, history.model.get_weights(),  training_loss, val_loss, nEpocs )

    def __init__( self, _val_loss_history, _train_loss_history, _initial_weights, _final_weights, _training_loss,  _val_loss, _nEpocs ):
        self.val_loss_history = _val_loss_history
        self.train_loss_history = _train_loss_history
        self.initial_weights = _initial_weights
        self.final_weights = _final_weights
        self.val_loss = _val_loss
        self.train_loss = _training_loss
        self.nEpocs = _nEpocs

    def valLossHistory(self):
        return self.val_loss_history

    def lossHistory(self):
        return self.train_loss_history

    def isMature(self):
        return self.val_loss < self.train_loss

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
            if isinstance(result, basestring):
                raise Exception( "A worker raised an Exception: " + result )
            elif result and result.isMature:
                if bestResult is None or result < bestResult:
                    bestResult = result
        return bestResult

class LearningModel:

    def __init__( self, inputDataset, trainingDataset,  **kwargs ):
        # type: (InputDataset, TrainingDataset) -> None
        self.inputs = inputDataset                  # type: PCDataset
        self.outputs = trainingDataset              # type: TrainingDataset
        self.batchSize = kwargs.get( 'batch', 50 )
        self.nEpocs = kwargs.get( 'epocs', 300 )
        self.validation_fraction = kwargs.get(  'vf', 0.1 )
        self.timeRange = kwargs.get(  'timeRange', None )
        self.hidden = kwargs.get( 'hidden', [16] )
        self.shuffle = kwargs.get('shuffle', False )
        self.weights = kwargs.get( 'weights', None )
        self.lr = kwargs.get( 'lrate', 0.01 )
        self.momentum = kwargs.get( 'momentum', 0.0 )
        self.decay = kwargs.get( 'decay', 0.0 )
        self.nesterov = kwargs.get( 'nesterov', False )
        self.max_loss = kwargs.get( 'max_loss', sys.float_info.max )
        self.activation = kwargs.get( 'activation', 'relu' )
        self.timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
        self.projectName = self.inputs.getName()
        self.logDir = os.path.expanduser("~/results/logs/{}".format( self.projectName + "_" + self.timestamp ) )
        self.inputData = self.inputs.getEpoch()
        self.dates, self.outputData = self.outputs.getEpoch()
        self.tensorboard = TensorBoard( log_dir=self.logDir, histogram_freq=0, write_graph=True )
        self.bestFitResult = None # type: FitResult
        self.weights_template = self.createSequentialModel().get_weights()

    def execute( self, nIterations=1, procIndex=-1 ):
        # type: () -> FitResult
        self.reseed()
        for iterIndex in range(nIterations):
            model = self.createSequentialModel()
            initial_weights = model.get_weights()
            history = model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=self.shuffle, callbacks=[self.tensorboard], verbose=0 )  # type: History
            self.updateHistory( history, initial_weights, iterIndex, procIndex )
        return self.bestFitResult

    def reseed(self):
        seed = random.randint(0, 2 ** 32 - 2)
        np.random.seed(seed)

    def getRandomWeights(self):
        self.reseed()
        new_weights = []
        for wIndex in range(len(self.weights_template)):
            wshape = self.weights_template[wIndex].shape
            if wIndex % 2 == 1:     new_weights.append(  2 * np.random.random_sample( wshape ) - 1 )
            else:                   new_weights.append( np.zeros( wshape, dtype=float ) )
        return new_weights

    def getInitialModel(self, fitResult):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()
        model.set_weights( fitResult.initial_weights )
        return model

    def getFittedModel( self, fitResult ):
        # type: (FitResult) -> Model
        model = self.createSequentialModel()  # type: Sequential
        model.set_weights( fitResult.initial_weights )
        model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=fitResult.nEpocs, validation_split=self.validation_fraction, shuffle=self.shuffle, verbose=0 )  # type: History
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
        model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=self.shuffle, verbose=0 )  # type: History
        return model

    def createSequentialModel( self ):
        # type: () -> Sequential
        model = Sequential()
        nHidden = len(self.hidden)
        nOutputs = self.outputs.getOutputSize()
        nInputs = self.inputs.getInputDimension()

        for hIndex in range(nHidden):
            if hIndex == 0:
                model.add(Dense(units=self.hidden[hIndex], activation=self.activation, input_dim=nInputs))
            else:
                model.add(Dense(units=self.hidden[hIndex], activation=self.activation))
        output_layer = Dense(units=nOutputs) if nHidden else Dense(units=nOutputs, input_dim=nInputs)
        model.add( output_layer )
        sgd = SGD( lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov )
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        if self.weights is not None: model.set_weights(self.weights)
        return model

    def updateHistory( self, history, initial_weights, iterIndex, procIndex ):
        # type: (History, list[np.ndarray], int, int) -> History
        val_loss = history.history['val_loss']
        training_loss = history.history['loss']
        current_min_loss_val = sys.float_info.max
        current_min_loss_index = -1
        for i in xrange( len(val_loss) ):
            vloss, tloss = val_loss[i], training_loss[i]
            if tloss < self.max_loss and vloss < current_min_loss_val:
               current_min_loss_val =  vloss
               current_min_loss_index = i
        if current_min_loss_index > 0:
            if not self.bestFitResult or ( current_min_loss_val < self.bestFitResult.val_loss ):
                self.bestFitResult = FitResult.new( history, initial_weights, training_loss[-1], current_min_loss_val, current_min_loss_index )
            if procIndex < 0:   print "Executed iteration {0}, current loss = {1}  (NE={2}), best loss = {3} (NE={4})".format(iterIndex, current_min_loss_val, current_min_loss_index, self.bestFitResult.val_loss, self.bestFitResult.nEpocs )
            else:               logging.info( "PROC[{0}]: Iteration {1}, val_loss = {2}, val_loss index = {3}, min val_loss = {4}".format( procIndex, iterIndex, current_min_loss_val, current_min_loss_index, self.bestFitResult.val_loss ) )
        return history

    def plotPrediction( self, fitResult, title, **kwargs ):
        # type: (FitResult, str) -> None
        print "Plotting result: " + title
        print " ---> NEpocs = {0}, loss = {1}, loss history = {2}".format(fitResult.nEpocs, fitResult.val_loss, str(fitResult.val_loss_history))
        model1 = self.getFittedModel(fitResult)
        prediction1 = model1.predict( self.inputData )  # type: np.ndarray
        targetNames = self.outputs.targetNames()
        for iPlot in range(prediction1.shape[1]):
            plt.title( targetNames[iPlot] + ": " + title )
            plt.plot_date(self.dates, prediction1[:,iPlot], "-", label="prediction: fitted" )
            plt.plot_date(self.dates, self.outputData[:,iPlot], "--", label="training data" )
            plt.legend()
            plt.show()

    def plotPerformance( self, fitResult, title, **kwargs ):
        # type: (FitResult, str) -> None
        valLossHistory = fitResult.valLossHistory()
        trainLossHistory = fitResult.lossHistory()
        plt.title(title)
        print "Plotting result: " + title
        x = range( len(valLossHistory) )
        plt.plot( x, valLossHistory, "--", label="Validation Loss" )
        plt.plot( x, trainLossHistory, "--", label="Training Loss" )
        plt.legend()
        plt.show()

    @classmethod
    def serial_execute(cls, lmodel_factory, nInterations, procIndex=-1, resultQueue=None):
        # type: (object(), int, int, mp.Queue) -> Union[FitResult,str]
        try:
            learningModel = lmodel_factory()  # type: LearningModel
            result = learningModel.execute( nInterations, procIndex )
            if resultQueue is not None: resultQueue.put( result )
            return result
        except Exception as err:
            msg = "PROC[{0}]: {1}".format( procIndex, str(err) )
            logging.error( msg )
            logging.error(traceback.format_exc())
            if resultQueue is not None: resultQueue.put(msg)
            return msg

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

            print " ** Executing {0} iterations on each of {1} processors for a total of {2} iterations, logging to {3}".format(nIterPerProc,nProc,nIterPerProc*nProc,LOG_FILE)

            for iProc in range(nProc):
                print " ** Executing process " + str(iProc) + ": NIterations: " + str(nIterPerProc)
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

