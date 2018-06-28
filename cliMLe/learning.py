from keras.callbacks import History

class FitResult:

    def __init__( self, _history, _val_loss, _nEpocs ):
        self.history = _history
        self.val_loss = _val_loss
        self.nEpocs = _nEpocs

    def valLossSeries(self):
        return self.history.history['val_loss']

    def model(self):
        return self.history.model