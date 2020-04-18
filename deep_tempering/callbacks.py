"""Wrappers for Keras' callbacks."""
import copy
import random
import functools
import collections

import tensorflow as tf
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np

make_logs = functools.partial(cbks.make_logs)

def get_progbar(model):
  """Get Progbar."""
  stateful_metric_names = model.metrics_names[model.n_replicas:]
  return cbks.ProgbarLogger('samples', stateful_metric_names)

def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        samples=None,
                        epochs=None,
                        steps_per_epoch=None,
                        verbose=1,
                        mode=ModeKeys.TRAIN):
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
  if isinstance(callbacks, (cbks.CallbackList, CallbackListWrapper)):
    if mode == ModeKeys.TEST:
      callbacks.set_test_progbar(get_progbar(model), verbose=verbose)
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  ################# testing #################
  x = np.random.normal(0, 1, (10, 2))
  y = np.random.randint(0, 2, (10, 1))
  exchange_data = (x, y)
  swap_step = 4
  model.exchange_callback = MetropolisExchangeCallback(exchange_data, swap_step)
  callbacks += [model.exchange_callback]
  ###################################################
  if mode == ModeKeys.TRAIN:
    model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]

  
  callback_list = CallbackListWrapper(callbacks)

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
  if verbose:
    progbar = get_progbar(model)
    callback_list.set_progbar(progbar, verbose=verbose)
  if mode == ModeKeys.TRAIN:
    # `global_step` is incremented each gradient update step
    callback_list.model.global_step = 0


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


class CallbackListWrapper(cbks.CallbackList):
  """Wrapper for CallbackList instance.

  Before tensorflow2.2 progress bar is implemented separetely.
  In tensorflow 2.2 the progbar callback could be taken care of
  within the `CallbackList` instance. Currently, it is implemented
  separately. See the following commit for more:
  https://github.com/tensorflow/tensorflow/commit/10666c59dd4858645d1b03ce01f4450da80710ec
  """
  def __init__(self, *args, **kwargs):
    super(CallbackListWrapper, self).__init__(*args, **kwargs)
    self.progbar = None
    self._train_progbar = None

  def set_progbar(self, progbar, verbose=0):
    self.progbar = progbar
    self.progbar.params = self.params
    self.progbar.params['verbose'] = verbose

  def set_test_progbar(self, progbar, verbose):
    if self.progbar is not None:
      self._train_progbar = self.progbar
      self.progbar = progbar
      self.progbar.params = self.params
      self.progbar.params['verbose'] = verbose

  def maybe_exchange(self):
    model = self.model
    exchange_callback = getattr(self.model, 'exchange_callback', None)
    if exchange_callback is not None and exchange_callback.should_exchange:
      # compute exchange losses and call user defined `exchange()` function
      x, y = exchange_callback.exchange_data
      losses = model.evaluate(x, y, verbose=0)
      losses = [l[0] for l in losses]
      exchange_callback.exchange(losses)


  def _call_begin_hook(self, mode):
    super()._call_begin_hook(mode)
    if self.progbar is not None:
      self.progbar.on_train_begin()

  def _call_epoch_hook(self, mode, hook_name, epoch, epoch_logs):
    if hook_name == 'begin':
      return self._on_epoch_begin(epoch, epoch_logs, mode=mode)
    elif hook_name == 'end':
      return self._on_epoch_end(epoch, epoch_logs, mode=mode)

  def _on_epoch_begin(self, epoch, epoch_logs, mode=None):
    if mode == ModeKeys.TRAIN: 
      super().on_epoch_begin(epoch, epoch_logs)
    if self.progbar is not None:
      self.progbar.on_epoch_begin(epoch, epoch_logs)

  def _on_epoch_end(self, epoch, epoch_logs, mode=None):
    # Epochs only apply to `fit`.
    if mode == ModeKeys.TRAIN: 
      super().on_epoch_end(epoch, epoch_logs)
    if self.progbar is not None:
      self.progbar.on_epoch_end(epoch, epoch_logs)

  def _call_batch_hook(self, mode, hook_name, batch_index, batch_logs):
    super()._call_batch_hook(mode, hook_name, batch_index, batch_logs)
    if self.progbar is not None:
      if hook_name == 'begin':
        self.progbar.on_batch_begin(batch_index, batch_logs)
      elif hook_name == 'end':
        self.progbar.on_batch_end(batch_index, batch_logs)

    # increment global step `on_batch_end()` and try to exchange
    if mode == ModeKeys.TRAIN and hook_name == 'end':      
      self.model.global_step += 1
      self.maybe_exchange()

  def _call_end_hook(self, mode):
    # Remove test progbar if it was created. Set train progress bar
    # if it was previously existed. It happens at the end of the
    # validation loop during training iterations
    if mode == ModeKeys.TEST:
      test_progbar = self.progbar
      self.progbar = self._train_progbar
      del test_progbar
    super()._call_end_hook(mode)


class BaseExchangeCallback(tf.keras.callbacks.Callback):
  def __init__(self, exchange_data, swap_step, burn_in=None):
    super(BaseExchangeCallback, self).__init__()
    self.exchange_data = exchange_data
    self.swap_step = swap_step
    self.burn_in = burn_in or 1

  def log_exchange(self, losses, hpname=None, exchange_pair=None, proba=None):

    # first log
    if not hasattr(self, 'per_replica_logs'):
      self.per_replica_logs = _init_per_replica_logs(self)

    if not hasattr(self, 'progress_logs'):


    # log exchangable hyperparams
    hpspace = self.model.hpspace
    n_replicas = self.model.n_replicas
    for i in range(n_replicas):
      for name in hpspace.hyperparameters_names:
        self.per_replica_logs[i][name].append(hpspace.hpspace[i][name])

      # log exchange losses
      self.per_replica_logs[i]['loss'].append(losses[i])


  @property
  def should_exchange(self):
    global_step = self.model.global_step
    return (global_step > self.burn_in
            and global_step % self.swap_step == 0)

  @property
  def ordered_hyperparams(self):
    result = {}
    hpspace = self.model.hpspace
    for hpname in hpspace.hyperparameters_names:
      result[hpname] = hpspace.get_ordered_hparams(hpname)
    return result

  def get_ordered_losses(self, logs):
    all_losses_keys = [l for l in logs.keys() if l.startswith('loss')]
    all_losses_keys.sort(key=self._metrics_sorting_key)
    res = [(name, logs[name]) for name in all_losses_keys]
    return res

  def _metrics_sorting_key(self, metric_name):
    """Key for sorting indexed metrics names.

    For example, losses indexed with with one index level
    (loss_1, loss_2, etc) come before losses with two index
    levels (loss_1_0, loss_1_1).
    """
    splitted = metric_name.split('_')
    # `index_level` is the level of underscore indices in the metric name
    # For example, `index_level` of `loss_1_2` is 2. 
    index_level = 0
    indices = []
    for item in reversed(splitted):
      if item.isdigit():
        index_level += 1
        indices.append(item)
      else:
        break
    return int(str(index_level) + ''.join(reversed(indices)))

class MetropolisExchangeCallback(BaseExchangeCallback):
  """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
  def __init__(self, exchange_data, swap_step, burn_in=None):
    super(MetropolisExchangeCallback, self).__init__(
        exchange_data, swap_step, burn_in)


  def exchange(self, losses, hpname=None, exchange_pair=None, proba=None):
    hp = self.ordered_hyperparams
    n_replicas = self.model.n_replicas
    # pick random hyperparameter to exchange
    hpname = hpname or random.choice(list(hp.keys()))
    hyperparams = [h[1] for h in hp[hpname]]

    # pick random replica pair to exchange
    exchange_pair = exchange_pair or np.random.randint(1, n_replicas)
    i = exchange_pair
    j = exchange_pair - 1

    # compute betas
    if 'dropout' in hpname:
      beta_i = (1. - hyperparams[i]) / hyperparams[i]
      beta_j = (1. - hyperparams[j]) / hyperparams[j]
    else:
      # learning rate
      beta_i = 1. / hyperparams[i]
      beta_j = 1. / hyperparams[j]

    # beta_i - beta_j is expected to be negative
    proba = proba or min(np.exp(
        coeff * (losses[i] - losses[j]) * (beta_i - beta_j)), 1.)

    if np.random.uniform() < proba:
      self.model.hpspace.swap_between(i, j, hpname)

    super().log_exchange(losses, hpname=hpname, exchange_pair=exchange_pair, proba=proba)

def _init_per_replica_logs(callback):
  """Creates a dict that stores exchange logs per replica."""
  model = callback.model
  hpspace = model.hpspace
  n_replicas = model.n_replicas
  logs_names = hpspace.hyperparameters_names + ['loss']
  result = {
      i: {name: [] for name in logs_names}
      for i in range(n_replicas)
  }

  return result