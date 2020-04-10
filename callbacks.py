"""Wrappers for Keras' callbacks."""
import copy
import functools

import tensorflow as tf
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.utils.mode_keys import ModeKeys

make_logs = functools.partial(cbks.make_logs)

# class CallbackList(cbks.CallbackList):

#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)

#   def _call_begin_hook(self, mode):
#     super()._call_begin_hook(mode)

#     if tf.__version__ <= '1.15.2':
#       progbar = self.callbacks[2]
#       if isinstance(progbar, cbks.ProgbarLogger):
#         progbar.on_train_begin()

#   def _call_batch_hook(self, mode, )

def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        mode=ModeKeys.TRAIN,
                        count_mode='samples'):
  """Configures callbacks for use in various training loops.

  It is just a reimplementation of `set_callback_parameters()`
  from `tensorflow.python.keras.callbacks` that modified to support
  `EnsembleModel` instead of `tf.keras.Model()`.

  Args:
    callbacks: List of Callbacks.
    model: EnsembleModel being trained.
    do_validation: Whether or not validation loop will be run.
    batch_size: Number of samples per batch.
    epochs: Number of epoch to train.
    steps_per_epoch: Number of batches to run per training epoch.
    samples: Number of training samples.
    verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
    count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
    mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
      Which loop mode to configure callbacks for.

  Returns:
      Instance of `CallbackList` used to control all Callbacks.
  """

  # Check if callbacks have already been configured.
  if isinstance(callbacks, cbks.CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == ModeKeys.TRAIN:
    model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]

  # implement a new progress bar here
  if verbose:
    progbar = cbks.ProgbarLogger(count_mode)
    callbacks.append(progbar)
  callback_list = cbks.CallbackList(callbacks)

  # Set callback model
  callback_model = model #._get_callback_model()  # pylint: disable=protected-access
  callback_list.set_model(callback_model)

  set_callback_parameters(
      callback_list,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=samples,
      verbose=verbose,
      mode=mode)

  callback_list.model.stop_training = False

  # if tf.__version__ <= '1.15.2':
  #   progbar.param = callback_list.params
  #   progbar.params['verbose'] = verbose

  return callback_list


def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
  """Sets callback parameters.

  Args:
    callback_list: CallbackList instance.
    model: EnsembleModel being trained.
    do_validation: Whether or not validation loop will be run.
    batch_size: Number of samples per batch.
    epochs: Number of epoch to train.
    steps_per_epoch: Number of batches to run per training epoch.
    samples: Number of training samples.
    verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
    mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
      Which loop mode to configure callbacks for.
  """
  for cbk in callback_list:
    if isinstance(cbk, (cbks.BaseLogger, cbks.ProgbarLogger)):
      cbk.stateful_metrics = model.metrics_names[model.n_replicas:]  # Exclude `loss`

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != ModeKeys.PREDICT and hasattr(model, 'metrics_names'):
    callback_metrics = copy.copy(model.metrics_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in model.metrics_names]
  callback_params = {
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  }
  callback_list.set_params(callback_params)
