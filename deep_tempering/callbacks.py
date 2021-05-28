"""Wrappers for Keras' callbacks."""
import os
import copy
import random
import functools
import collections
import json

import tensorflow as tf
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np

from deep_tempering import training_utils

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
                        ):
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
    if not any(isinstance(c, MonitorOptimalModelCallback) for c in callbacks):
      callbacks += [MonitorOptimalModelCallback()]
    # add default exchange callback to the `callbacks_list` (if not there)
    if not any(isinstance(c, BaseExchangeCallback) for c in callbacks):
      callbacks += [MetropolisExchangeCallback(exchange_data)]
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
  )

  callback_list.model.stop_training = False
  if verbose:
    progbar = get_progbar(model)
    callback_list.set_progbar(progbar, verbose=verbose)

  # Set global step based on which the exchanges are scheduled.
  # This global step is incremented by `CallbackListWrapper` on each
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
                            ):
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
  def __init__(self, exchange_data, swap_step, burn_in):
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
    self.burn_in = burn_in

  @property
  def exchangable(self):
    return self.swap_step is not None and self.exchange_data is not None

  def evaluate_metrics(self):
    """Evaluates losses and metrics on exchange dataset.
    TODO: Add sample weights and class weight option.
    """
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
            and (global_step - self.burn_in) % self.swap_step == 0)

  def exchange(self):
    """This method must be implemented in subclasses.

    This function is called once on the beginning of training to
    log initial values of hyperparameters and then it is called
    every `swap_step` steps.
    """
    raise NotImplementedError()

  def _safe_exchange(self, *args, **kwargs):
    if not self.exchangable:
      return

    self.exchange(*args, **kwargs)


  @property
  def ordered_hyperparams(self):
    result = {}
    hpspace = self.model.hpspace
    for hpname in hpspace.hyperparameters_names:
      result[hpname] = hpspace.get_ordered_hparams(hpname)
    return result

  def get_ordered_losses(self, logs):
    return get_ordered_metrics(logs, 'loss')

class PBTExchangeCallback(BaseExchangeCallback):
  """Exchanges of parameters based on PBT scheduling.

  See: Population Based Training of Neural Networks
       https://arxiv.org/abs/1711.09846
  NOTES:
    * Replica/Worker and Ensemble/Population are used interchangeably
      in the code and docs.
    * `exploit()` and `explore()` methods correspond to the ones in the
      original paper, except that perform the actions for the entire
      population (and not for single individual replica).
  """

  def __init__(self,
               exchange_data,
               swap_step,
               burn_in=None,
               explore_weights=False,
               explore_hyperparams=True,
               weight_dist_fn=None,
               hyperparams_dist=None):
    """Instantiates a new `PBTExchangeCallback` instance.

    Args:
      weight_dist_fn: A function that given shape returns numpy array
        of random values that are added to the to the weights. E.g.
        `weight_dist_fn = functools.partial(np.random.normal, 0, 0.1)`
      hyperparams_dist: A dictionary that maps hyperpamater name to a
        function that returns random value by which the respective
        hyperparameter is perturbed. For example:
    """
    self.should_explore_weights = explore_weights
    self.should_explore_hyperparams = explore_hyperparams
    self.weight_dist_fn = (weight_dist_fn
                           or functools.partial(np.random.normal, 0, 0.1))
    self.hyperparams_dist = hyperparams_dist

    super(PBTExchangeCallback, self).__init__(exchange_data, swap_step, burn_in)

  def exploit_and_explore(self, **kwargs):
    """Decides whether  the worker should abandon the current solution.

    Given performance of the whole population, can decide whether the
    worker should abandon the current solution and instead focus on a
    more promising one; and `explore`, which given the current solution
    and hyperparameters proposes new ones to better explore the
    solution space.

    `exploit` could replace the current weights with the weights that
    have the highest recorded performance in the rest of the
    population, and `explore` could randomly perturb the
    hyperparameters with noise.

    In short, copies weights and hyperparams from optimal replica and
    perturbs them.
    """
    # `test_losses` is used for testing to verify the logic.
    losses = kwargs.get('test_losses', None) or self.evaluate_exchange_losses()
    optimal_replica_id = np.argmin(losses)

    optimal_weights = self.model.models[optimal_replica_id].trainable_variables

    # copy vars
    for rid in range(self.model.n_replicas):
      if rid != optimal_replica_id:
        if not tf.executing_eagerly():
          self.copy_weights(optimal_replica_id, rid)

          if self.should_explore_weights:
            self.explore_weights(rid)
          if self.should_explore_hyperparams:
            # copy hparams and perturb
            self.model.hpspace.copy_hyperparams(optimal_replica_id, rid)
            self.explore_hyperparams(rid)

        else:
          raise NotImplementedError()

    super().log_exchange_metrics(losses, optimal_replica=optimal_replica_id)

  def copy_weights(self, src_replica, dst_replica):
    """Copies variables from `src_replica` to `dst_replica`."""
    # print('copy_weights ---> src_replica:', src_replica, ', dst_replica:', dst_replica)
    src_model = self.model.models[src_replica]
    dst_model = self.model.models[dst_replica]
    src_vars = src_model.trainable_variables
    dst_vars = dst_model.trainable_variables
    sess = tf.compat.v1.keras.backend.get_session()
    for vsrc, vdst in zip(src_vars, dst_vars):
      np_vsrc = vsrc.eval(session=sess)
      vdst.load(np_vsrc, session=sess)

  def explore_weights(self, replica_id):
    """Perturbs weights of `replica_id` with noise.

    Args:
      replica_id: The ID of replica that needs to be perturbed.
    """
    weight_dist_fn = (self.weight_dist_fn
                      or functools.partial(np.random.normal, 0, 0.1))

    sess = tf.compat.v1.keras.backend.get_session()
    model = self.model.models[replica_id]
    for w in model.trainable_variables:
      shape = w.get_shape().as_list()
      value = sess.run(w)
      perturbed_value = value + self.weight_dist_fn(shape)
      w.load(perturbed_value, session=sess)

  def explore_hyperparams(self, replica_id):
    """Perturbs hyperparams of `replica_id`."""
    if self.hyperparams_dist is not None:
      for hpname, dist in self.hyperparams_dist.items():
        self.model.hpspace.perturb_hyperparams(
            replica_id, hpname, dist)

  def copy_hyperparams(self, src_replica, dst_replica):
    """Copies variables from `src_replica` to `dst_replica`."""
    hps = self.model.hpspace
    for hpname in hps.hpspace[0]:
      hps.hpspace[dst_replica][hpname] = hps.hpspace[src_replica][hpname]

  def exchange(self, *args, **kwargs):
    self.exploit_and_explore(*args, **kwargs)


class MetropolisExchangeCallback(BaseExchangeCallback):
  """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
  def __init__(self, exchange_data, swap_step=1, burn_in=1, coeff=1.):
    super(MetropolisExchangeCallback, self).__init__(exchange_data, swap_step, burn_in)
    self.coeff = coeff

  def exchange(self, **kwargs):
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

    losses = self.evaluate_exchange_losses()

    hyperparams = [h[1] for h in hp[hpname]]
    replicas_ids = [h[0] for h in hp[hpname]]

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
    delta = self.coeff * (losses[replicas_ids[i]] - losses[replicas_ids[j]]) * (beta_i - beta_j)
    proba = min(np.exp(delta), 1.)

    if np.random.uniform() < proba:
      swaped = 1
      self.model.hpspace.swap_between(replicas_ids[i], replicas_ids[j], hpname)
    else:
      swaped = 0

    if getattr(self, 'exchange_logs', None):
      n_attempts = len(self.exchange_logs['proba']) + 1
      n_swaps = self.exchange_logs['swaped'].count(1) + swaped
      accpt_ratio = n_swaps / n_attempts
    else:
      accpt_ratio = swaped

    super().log_exchange_metrics(losses,
                                 proba=proba,
                                 hpname=hpname,
                                 swaped=swaped,
                                 accept_ratio=accpt_ratio,
                                 delta=delta,
                                 exchange_pair=[replicas_ids[i], replicas_ids[j]])


class MonitorOptimalModelCallback(tf.keras.callbacks.Callback):
  """Monitors optimal keras' model.

  At the end of each epoch stores the optimal keras model based on value
  we are metric value being monitored. This callback is added automatically
  if any subclass of this instance is not passed explitly within list
  of callbacks during `fit()`.
  """
  def __init__(self, monitor='val_loss', path=None, name=None):
    """Instantiatiates a  new `MonitorOptimalModelCallback` instance.

    Args:
      monitor: A value of a metric to monitor.
      path: A directory where to store the otimal model. By default,
        stores at current working directory in the `.deep_tempering_model`.
    """
    self.monitor = monitor
    self.path = path or training_utils.LOGS_PATH
    self.name = name or 'optimal_model.h5'

  def on_epoch_end(self, epoch, logs):
    # get metrics ordered by replica_id
    monitored_metrics = get_ordered_metrics(logs, self.monitor)

    # if empty there is nothing to monitor, then just return
    if not monitored_metrics:
      return

    # add replica id to the the monitored metrics
    monitored_metrics = [m + (i,) for i, m in enumerate(monitored_metrics)]
    min_or_max = training_utils.min_or_max_for_metric(self.monitor)
    if min_or_max == 'min':
      fn = min
    else:
      fn = max

    optimal = fn(monitored_metrics, key=lambda x: x[1])
    optimal_replica_id = optimal[2]
    optimal_model = self.model.models[optimal_replica_id]
    if not os.path.exists(self.path):
      os.makedirs(self.path)

    # store weights and hyperparams of the optimal model
    optimal_model.save_weights(os.path.join(self.path, self.name))
    with open(os.path.join(self.path, 'hyperparams.json'), 'w') as fo:
      json.dump(self.model.hpspace.hpspace[optimal_replica_id], fo, indent=2)


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
