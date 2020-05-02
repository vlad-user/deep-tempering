"""Wrappers for Keras' callbacks."""
import copy
import random
import functools
import collections

import tensorflow as tf
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np

make_logs = cbks.make_logs

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
                        mode=ModeKeys.TRAIN,
                        exchange_data=None,
                        swap_step=None,
                        burn_in=None):
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
  if mode == ModeKeys.TRAIN:
    model.history = cbks.History()
    if not any(isinstance(c, BaseExchangeCallback) for c in callbacks):
      callbacks += [MetropolisExchangeCallback(exchange_data, swap_step, burn_in)]
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
      mode=mode,
      swap_step=swap_step,
      burn_in=burn_in)

  callback_list.model.stop_training = False
  if verbose:
    progbar = get_progbar(model)
    callback_list.set_progbar(progbar, verbose=verbose)

  # Set global step based on which exchanges are scheduled.
  # This global step is incremented by CallbackListWrapper on each
  # training batch end.
  if mode == ModeKeys.TRAIN:
    model.global_step = 0

  return callback_list

def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN,
                            swap_step=None,
                            burn_in=None):
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
      'swap_step': swap_step,
      'burn_in': burn_in
  }
  callback_list.set_params(callback_params)


class CallbackListWrapper(cbks.CallbackList):
  """Wrapper for CallbackList instance.

  Before tensorflow2.2 progress bar is implemented separetely.
  In tensorflow 2.2 the progbar callback could be taken care of
  within the `CallbackList` instance. Currently, it is implemented
  separately. See the following commit for more:
  https://github.com/tensorflow/tensorflow/commit/10666c59dd4858645d1b03ce01f4450da80710ec

  For details about the functions overriden here see
  `tf.keras.callbacks.CallbackList`.
  """
  def __init__(self, *args, **kwargs):
    super(CallbackListWrapper, self).__init__(*args, **kwargs)
    self.progbar = None
    self._train_progbar = None

  def set_progbar(self, progbar, verbose=0):
    """Sets progress bar."""
    self.progbar = progbar
    self.progbar.params = self.params
    self.progbar.params['verbose'] = verbose

  def set_test_progbar(self, progbar, verbose=0):
    """Sets testing progress bar without losing ref for training one."""
    if self.progbar is not None:
      self._train_progbar = self.progbar
      self.progbar = progbar
      self.progbar.params = self.params
      self.progbar.params['verbose'] = verbose

  def _call_begin_hook(self, mode):
    super()._call_begin_hook(mode)
    if self.progbar is not None:
      self.progbar.on_train_begin()

    # exchange on the beginning of training so we could log
    # initial state hyperparameter state of each replica.
    if mode == ModeKeys.TRAIN:
      for callback in self.callbacks:
        if isinstance(callback, BaseExchangeCallback):
          callback._safe_exchange()

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
        # increment global step during train mode
        if mode == ModeKeys.TRAIN:
          self.model.global_step += 1
          # attempt exchanges
          for callback in self.callbacks:
            if (isinstance(callback, BaseExchangeCallback)
                and callback.should_exchange()):
              callback._safe_exchange()
        self.progbar.on_batch_end(batch_index, batch_logs)

  def _call_end_hook(self, mode):
    if mode == ModeKeys.TEST:
      test_progbar = self.progbar
      self.progbar = self._train_progbar
      del test_progbar

    super()._call_end_hook(mode)

    # Attach exchange logs to the HistoryCallback.
    # Assuming there only one Exchange callback.
    # In case of multiple exchange callbacks they could be accessed
    # through exchange callbacks themselfs.
    for callback in self.callbacks:
      if isinstance(callback, BaseExchangeCallback):
        self.model.history.exchange_history = getattr(callback, 'exchange_logs', None)
        break

class BaseExchangeCallback(tf.keras.callbacks.Callback):
  """Base class for exchanges.
  
  You never use this class directly, but instead instantiate one of
  its subclasses such as `dt.callbacks.MetropolisExchangeCallback`.
  """
  def __init__(self, exchange_data, swap_step, burn_in=None):
    """Initializes a new `BaseExchangeCallback` instance.

    Args:
      exchange_data: A list or tuple of (x, y) data (same structure
        as `validation_data` argument in `keras.model.fit()`.)
      swap_step: A step at wich the exchange is performed.
      burn_in: As step before which the exchanges are not perfomed.
    """
    super(BaseExchangeCallback, self).__init__()
    self.exchange_data = exchange_data
    self.swap_step = swap_step
    self.burn_in = burn_in or 1

  @property
  def exchangable(self):
    return self.swap_step is not None and self.exchange_data is not None

  def evaluate_metrics(self):
    """Evaluates losses and metrics on exchange dataset."""
    if not self.exchangable:
      return []
    x, y = self.exchange_data
    metrics = self.model.evaluate(x, y, verbose=0)
    return metrics

  def evaluate_exchange_losses(self):
    """Evaluates losses on exchange dataset."""
    if not self.exchangable:
      return []
    metrics = self.evaluate_metrics()
    return metrics[:self.model.n_replicas]

  def log_exchange_metrics(self, losses, **kwargs):
    """Logs metrics related to exchanges.

    Args:
      losses: A list of exchange losses.
      kwargs: Any metric that user wishes to log.
    """
    exchange_logs = (getattr(self, 'exchange_logs', None)
                     or _init_exchange_logs(self, kwargs))

    # log losses
    losses_names = self.model.metrics_names[:self.model.n_replicas]
    for i, loss_name in enumerate(losses_names):
      exchange_logs[loss_name].append(losses[i])

    # log the rest of the metrics (in kwargs)
    for k, v in kwargs.items():
      exchange_logs[k].append(v)

    # log hyper parameters
    for i in range(self.model.n_replicas):
      for name in self.model.hpspace.hyperparameters_names:
        exchange_logs[i][name].append(self.model.hpspace.hpspace[i][name])

    # log global step
    exchange_logs['step'].append(self.model.global_step)

  def should_exchange(self):
    """Whether to exchange based on swap step and burn in period."""
    global_step = self.model.global_step
    return (self.exchangable
            and global_step >= self.burn_in
            and self.swap_step is not None
            and global_step % self.swap_step == 0)

  def exchange_hyperparams(self):
    """This method must be implemented in subclasses.

    This function is called once on the beginning of training to
    log initial values of hyperparameters and then it is called
    every `swap_step` steps.
    """
    raise NotImplementedError()

  def _safe_exchange(self, *args, **kwargs):
    if not self.exchangable:
      return

    self.exchange_hyperparams(*args, **kwargs)


  @property
  def ordered_hyperparams(self):
    result = {}
    hpspace = self.model.hpspace
    for hpname in hpspace.hyperparameters_names:
      result[hpname] = hpspace.get_ordered_hparams(hpname)
    return result

  def get_ordered_losses(self, logs):
    return get_ordered_metrics(logs, 'loss')

class MetropolisExchangeCallback(BaseExchangeCallback):
  """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
  def __init__(self, exchange_data, swap_step, burn_in=None):
    super(MetropolisExchangeCallback, self).__init__(exchange_data, swap_step, burn_in)

  def exchange_hyperparams(self, **kwargs):
    """Exchanges hyperparameters between adjacent replicas.

    This function is called once on the beginning of training to
    log initial values of hyperparameters and then it is called
    every `swap_step` steps.
    """
    # pick random hyperparameter to exchange
    hp = self.ordered_hyperparams
    hpname = kwargs.get('hpname', random.choice(list(hp.keys())))
    # pick random replica pair to exchange
    n_replicas = self.model.n_replicas
    exchange_pair = kwargs.get('exchange_pair', np.random.randint(1, n_replicas))
    coeff = kwargs.get('coeff', 1.)
    losses = self.evaluate_exchange_losses()

    hyperparams = [h[1] for h in hp[hpname]]

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
    proba = min(np.exp(
        coeff * (losses[i] - losses[j]) * (beta_i - beta_j)), 1.)

    if np.random.uniform() < proba:
      swaped = 1
      self.model.hpspace.swap_between(i, j, hpname)
    else:
      swaped = 0

    super().log_exchange_metrics(losses, proba=proba, hpname=hpname,
                                 swaped=swaped)

def _init_exchange_logs(callback, metrics_dict=None):
  """Initializes `dict` that stores logs from replica exchanges."""
  metrics_dict = metrics_dict or {}
  hpspace = callback.model.hpspace
  logs_names = hpspace.hyperparameters_names
  result = {
      i: {name: [] for name in logs_names}
      for i in range(callback.model.n_replicas)
  }
  result.update({m: [] for m in metrics_dict})
  # add losses
  losses_names = callback.model.metrics_names[:callback.model.n_replicas]
  result.update({loss_name: [] for loss_name in losses_names})

  # global step
  result['step'] = []

  callback.exchange_logs = result
  return result

def _metrics_sorting_key(metric_name):
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

def get_ordered_metrics(logs, metric_name='loss'):
  all_losses_keys = [l for l in logs.keys() if l.startswith(metric_name)]
  all_losses_keys.sort(key=_metrics_sorting_key)
  res = [(name, logs[name]) for name in all_losses_keys]
  return res