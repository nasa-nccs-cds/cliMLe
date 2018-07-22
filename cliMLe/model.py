from keras.models import Sequential, Model
import warnings
import copy
import numpy as np
from keras.engine import training_arrays
from scipy.sparse import issparse
from keras import backend as K
from keras import callbacks as cbks
from keras.utils.generic_utils import slice_arrays
from keras.engine.training_utils import batch_shuffle
from keras.engine.training_utils import check_num_samples

class CustomSequentialModel(Sequential):


    def __init__(self, layers=None, name=None):
        super(CustomSequentialModel, self).__init__(layers=layers,name=name)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        """Trains the model for a given number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
                If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data
                (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        # Backwards compatibility
        if batch_size is None and steps_per_epoch is None:
            batch_size = 32
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` has been renamed `epochs`.', stacklevel=2)
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
        if x is None and y is None and steps_per_epoch is None:
            raise ValueError('If fitting from data tensors, you should specify the `steps_per_epoch` argument.')
        # Validate user data.
        x, y, sample_weights = self._standardize_user_data( x, y, sample_weight=sample_weight, class_weight=class_weight, batch_size=batch_size )
        # Prepare validation data.
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y,
                sample_weight=val_sample_weight,
                batch_size=batch_size)
            if self._uses_dynamic_learning_phase():
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            if any(K.is_tensor(t) for t in x):
                raise ValueError(
                    'If your data is in the form of symbolic tensors, '
                    'you cannot use `validation_split`.')
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(int(x[0].shape[0]) * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                slice_arrays(sample_weights, 0, split_at),
                slice_arrays(sample_weights, split_at))
            if self._uses_dynamic_learning_phase():
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_steps:
            do_validation = True
            if self._uses_dynamic_learning_phase():
                val_ins = [0.]

        # Prepare input arrays and training function.
        if self._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        f = self.train_function

        # Prepare display labels.
        out_labels = self.metrics_names

        if do_validation:
            self._make_test_function()
            val_f = self.test_function
            callback_metrics = copy.copy(out_labels) + [
                'val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)
            val_f = None
            val_ins = []

        # Delegate logic to `fit_loop`.
        return self.fit_loop( f, ins,
                                out_labels=out_labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=callbacks,
                                val_f=val_f,
                                val_ins=val_ins,
                                shuffle=shuffle,
                                callback_metrics=callback_metrics,
                                initial_epoch=initial_epoch,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps)

    def fit_loop( self, f, ins,
                 out_labels=None,
                 batch_size=None,
                 epochs=100,
                 verbose=1,
                 callbacks=None,
                 val_f=None,
                 val_ins=None,
                 shuffle=True,
                 callback_metrics=None,
                 initial_epoch=0,
                 steps_per_epoch=None,
                 validation_steps=None):
        """Abstract fit function for `f(ins)`.

        Assumes that f returns a list, labeled by out_labels.

        # Arguments
            model: Keras model instance.
            f: Keras function returning a list of tensors
            ins: List of tensors to be fed to `f`
            out_labels: List of strings, display names of
                the outputs of `f`
            batch_size: Integer batch size or None if unknown.
            epochs: Number of times to iterate over the data
            verbose: Verbosity mode, 0, 1 or 2
            callbacks: List of callbacks to be called during training
            val_f: Keras function to call for validation
            val_ins: List of tensors to be fed to `val_f`
            shuffle: Whether to shuffle the data at the beginning of each epoch
            callback_metrics: List of strings, the display names of the metrics
                passed to the callbacks. They should be the
                concatenation of list the display names of the outputs of
                 `f` and the list of display names of the outputs of `f_val`.
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run)
            steps_per_epoch: Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. Ignored with the default value of `None`.
            validation_steps: Number of steps to run validation for
                (only if doing validation from data tensors).
                Ignored with the default value of `None`.

        # Returns
            `History` object.
        """
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if (verbose and ins and
                    hasattr(ins[0], 'shape') and hasattr(val_ins[0], 'shape')):
                print('Train on %d samples, validate on %d samples' %
                      (ins[0].shape[0], val_ins[0].shape[0]))
        if validation_steps:
            do_validation = True
            if steps_per_epoch is None:
                raise ValueError('Can only use `validation_steps` '
                                 'when doing step-wise '
                                 'training, i.e. `steps_per_epoch` '
                                 'must be set.')
        elif do_validation:
            if steps_per_epoch:
                raise ValueError('Must specify `validation_steps` '
                                 'to perform validation '
                                 'when doing step-wise training.')

        num_train_samples = check_num_samples(ins, batch_size=batch_size, steps=steps_per_epoch, steps_name='steps_per_epoch')
        if num_train_samples is not None:
            index_array = np.arange(num_train_samples)

        self.history = cbks.History()
        _callbacks = [cbks.BaseLogger( stateful_metrics=self.stateful_metric_names) ]
        if verbose:
            if steps_per_epoch is not None:
                count_mode = 'steps'
            else:
                count_mode = 'samples'
            _callbacks.append( cbks.ProgbarLogger( count_mode, stateful_metrics=self.stateful_metric_names))
        _callbacks += (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(_callbacks)
        out_labels = out_labels or []

        # it's possible to callback a different model than itself
        # (used by Sequential models)
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks.set_model(callback_model)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': steps_per_epoch,
            'samples': num_train_samples,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics or [],
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False
        for cbk in callbacks:
            cbk.validation_data = val_ins

        # To prevent a slowdown,
        # we find beforehand the arrays that need conversion.
        feed = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
        indices_for_conversion_to_dense = []
        for i in range(len(feed)):
            if issparse(ins[i]) and not K.is_sparse(feed[i]):
                indices_for_conversion_to_dense.append(i)

        for epoch in range(initial_epoch, epochs):
            # Reset stateful metrics
            for m in self.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            if steps_per_epoch is not None:
                for step_index in range(steps_per_epoch):
                    batch_logs = {}
                    batch_logs['batch'] = step_index
                    batch_logs['size'] = 1
                    callbacks.on_batch_begin(step_index, batch_logs)
                    outs = f(ins)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(step_index, batch_logs)
                    if callback_model.stop_training:
                        break

                if do_validation:
                    val_outs = self.test_loop(self, val_f, val_ins, steps=validation_steps)
                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o
            else:
                if shuffle == 'batch':
                    index_array = batch_shuffle(index_array, batch_size)
                elif shuffle:
                    np.random.shuffle(index_array)

                batches = self.make_batches( num_train_samples, batch_size, len(index_array)-1 )
                for batch_index, (batch_start, batch_end) in enumerate(batches):
                    batch_ids = index_array[batch_start:batch_end]
                    try:
                        if isinstance(ins[-1], float):
                            # Do not slice the training phase flag.
                            ins_batch = slice_arrays(
                                ins[:-1], batch_ids) + [ins[-1]]
                        else:
                            ins_batch = slice_arrays(ins, batch_ids)
                    except TypeError:
                        raise TypeError('TypeError while preparing batch. '
                                        'If using HDF5 input data, '
                                        'pass shuffle="batch".')
                    batch_logs = {}
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = len(batch_ids)
                    callbacks.on_batch_begin(batch_index, batch_logs)
                    for i in indices_for_conversion_to_dense:
                        ins_batch[i] = ins_batch[i].toarray()

                    outs = f(ins_batch)
                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)
                    if callback_model.stop_training:
                        break

                    if batch_index == len(batches) - 1:  # Last batch.
                        if do_validation:
                            val_outs = self.test_loop(self, val_f, val_ins, batch_size=batch_size )
                            if not isinstance(val_outs, list):
                                val_outs = [val_outs]
                            # Same labels assumed.
                            for l, o in zip(out_labels, val_outs):
                                epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break
        callbacks.on_train_end()
        return self.history

    def make_batches( self, size, batch_size, max_index ):
        """Returns a list of batch indices (tuples of indices).

        # Arguments
            size: Integer, total size of the data to slice into batches.
            batch_size: Integer, batch size.

        # Returns
            A list of tuples of array indices.
        """
        num_batches = (size + batch_size - 1) // batch_size
        batches = []
        for i in range(num_batches):
            end = (i + 1) * batch_size
            batch = (i * batch_size, (i + 1) * batch_size ) if ( end <= max_index ) else (max_index - batch_size, max_index )
            batches.append( batch )
        return batches

    def test_loop(self, f, ins, batch_size=None, steps=None):
        """Abstract method to loop over some data in batches.

        # Arguments
            model: Keras model instance.
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size or `None`.
            verbose: verbosity mode.
            steps: Total number of steps (batches of samples)
                before declaring predictions finished.
                Ignored with the default value of `None`.

        # Returns
            Scalar loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """

        if hasattr(self, 'metrics'):
            for m in self.stateful_metric_functions:
                m.reset_states()
            stateful_metric_indices = [
                i for i, name in enumerate(self.metrics_names)
                if str(name) in self.stateful_metric_names]
        else:
            stateful_metric_indices = []

        num_samples = check_num_samples(ins, batch_size=batch_size, steps=steps, steps_name='steps')
        outs = []
        # To prevent a slowdown,
        # we find beforehand the arrays that need conversion.
        feed = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
        indices_for_conversion_to_dense = []
        for i in range(len(feed)):
            if issparse(ins[i]) and not K.is_sparse(feed[i]):
                indices_for_conversion_to_dense.append(i)

        if steps is not None:
            for step in range(steps):
                batch_outs = f(ins)
                if isinstance(batch_outs, list):
                    if step == 0:
                        for _ in enumerate(batch_outs):
                            outs.append(0.)
                    for i, batch_out in enumerate(batch_outs):
                        if i in stateful_metric_indices:
                            outs[i] = float(batch_out)
                        else:
                            outs[i] += batch_out
                else:
                    if step == 0:
                        outs.append(0.)
                    outs[0] += batch_outs

            for i in range(len(outs)):
                if i not in stateful_metric_indices:
                    outs[i] /= steps
        else:
            batches = self.make_batches(num_samples, batch_size, num_samples-1)
            index_array = np.arange(num_samples)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                if isinstance(ins[-1], float):
                    # Do not slice the training phase flag.
                    ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                else:
                    ins_batch = slice_arrays(ins, batch_ids)
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                batch_outs = f(ins_batch)
                if isinstance(batch_outs, list):
                    if batch_index == 0:
                        for batch_out in enumerate(batch_outs):
                            outs.append(0.)
                    for i, batch_out in enumerate(batch_outs):
                        if i in stateful_metric_indices:
                            outs[i] = batch_out
                        else:
                            outs[i] += batch_out * len(batch_ids)
                else:
                    if batch_index == 0:
                        outs.append(0.)
                    outs[0] += batch_outs * len(batch_ids)

            for i in range(len(outs)):
                if i not in stateful_metric_indices:
                    outs[i] /= num_samples
        if len(outs) == 1:
            return outs[0]
        return outs




