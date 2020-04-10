import itertools
from collections import abc

import tensorflow as tf
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np
import tqdm

import pt_train_utils as train_utils
import callbacks as cbks

class HPState:
  def __init__(self):
    self._attrs = {}

  def get_hparam(self, name, default_value=None):

    if name in self._attrs:
      raise ValueError('Hyper Params with name ', hp_name, 'already exists.')

    if default_value is None:
      hp = tf.compat.v1.placeholder(tf.float32, shape=(), name=name)
    else:
      hp = tf.compat.v1.placeholder_with_default(default_value,
                                                 shape=(),
                                                 name=name)
    self._attrs[name] = hp
    return hp

  def _get_hparam(self, name):
    if name in self._attrs:
      return self._attrs[name]

class HPSpaceState:
  """Represents the hyper-parameter state of all replicas.

  Serves as container for placeholders and actual values that
  are fed and updated during training.
  """
  def __init__(self, ensemble_model, hparams_dict):
    """Creates a new `HyperParamState` instance.
    ```python
    hparams_dict = {
        'learning_rate': np.linspace(0.001, 0.01, 6),
        'dropout_rate': np.linspace(0., 0.6, 6)
    }
    hps = HPSpaceState(hparams_dict)
    hps.hpspace
    # {0: {'learning_rate': 0.001, 'dropout_rate': 0.0},
    #  1: {'learning_rate': 0.0055000000000000005, 'dropout_rate': 0.3},
    #  2: {'learning_rate': 0.01, 'dropout_rate': 0.6}}
    ```
    """
    self.ensemble_model = ensemble_model
    hparams_dict = dict((k, list(v)) for k, v in hparams_dict.items())
    n_replicas = len(hparams_dict[hparams_dict.__iter__().__next__()])
    self.n_replicas = n_replicas
    self.hpspace = {
        i: {k: v[i] for k, v in hparams_dict.items()}
        for i in range(n_replicas)
    }

  def swap_between(self, replica_i, replica_j, name):
    hp_i = self.hpspace[replica_i][name]
    hp_j = self.hpspace[replica_j][name]
    self.hpspace[replica_j][name] = hp_i
    self.hpspace[replica_i][name] = hp_j

  def get_ordered_hparams(self, name):
    """Returns list of tuples of adjacent `(replica_id, hp_value)`."""
    hparams = [(i, self.hpspace[i][name]) for i in range(self.n_replicas)]
    hparams.sort(key=lambda x: x[1])
    return hparams

  def prepare_feed_tensors_and_values(self, training=True):

    n_replicas = len(self.hpspace)
    hpnames = list(self.hpspace[0].keys())
    current_hp_dict = {
        name: dict(self.get_ordered_hparams(name))
        for name in hpnames
    }

    hpstates = {
        i: self.ensemble_model._train_attrs[i]['hp_state']
        for i in range(n_replicas)
    }
    feed_dict = {}

    for i in range(n_replicas):
      for hpname in hpnames:
        if not training and hpname.startswith('dropout_rate'):
          value = 0.
        else:
          value = current_hp_dict[hpname][i]
        placeholder = hpstates[i]._get_hparam(hpname)

        assert placeholder is not None
        feed_dict[placeholder] = current_hp_dict[hpname][i]

    return feed_dict

class EnsembleModel:
  """Mimics the behaviour of `keras.Model` for ensemble PT training."""
  def __init__(self, model_builder):
    """Instantiates a new PTEnsemble instance."""
    if not callable(model_builder):
      raise TypeError("Expected callable `model_builder`.")

    self._model_builder_fn = model_builder
    self._is_compiled = False
    self.run_eagerly = False

    # Stores attributes such as losses, optimizers, states of hyperparameters
    # as dictionary {replica_id: {loss:..., optimizer:..,}}
    self._train_attrs = None

    # HPStateSpace instance could be instantiated only when exchange
    # hyperparameters are known - at `model.fit()`.
    self._hp_state_space = None

    # (maybe) should be protected  
    self.inputs = None
    self.outputs = None

  def compile(self,
              optimizer,
              loss,
              n_replicas,
              target_tensors=None):

    if any(arg is None for arg in (optimizer, loss, n_replicas)):
      raise ValueError('The arg is None')

    # validate losses
    # ...

    # validate optimizer
    # ...

    self.n_replicas = n_replicas

    # set config for `tf.keras.optimizers.Optimizer` subclasses
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
      config = optimizer.get_config()
      optimizer_config = {
          'class_name': config['name'],
          'config': config
    }
    elif isinstance(optimizer, str):
      optimizer_config = {
          'class_name': optimizer,
          'config': {}
      }
    elif (isinstance(optimizer, abc.Mapping)
          and 'class_name' in optimizer
          and 'config' in optimizer):
      optimizer_config = optimizer
    else:
      raise NotImplementedError("Not recognized optimizer.")

    train_attrs = {i: {} for i in range(self.n_replicas)}
    for i in range(self.n_replicas):

      # build model and the state of hyperparameters
      with tf.variable_scope('model_%d' % i):
        hp = HPState()
        train_attrs[i]['hp_state'] = hp
        # Each model has its own input placeholder, meaning that the
        # feed values are fed `n_replica` times. 
        # TODO: Implement this by using the same input for each one
        # of the model. This could reduce overhead of copying to
        # GPUs. In particular, the behaviour of
        # `prepare_input_output_tensors()` and values should be modified.
        model = self._model_builder_fn(hp)

      with tf.variable_scope('loss_%d' % i):
        outputs = model.outputs
        output_names = keras_train_utils.generic_output_names(outputs)
        loss_functions = keras_train_utils.prepare_loss_functions(loss,
                                                                  output_names)

      # Set placeholders instead of actual values for each possible
      # hyperparameterd of the optimizer. All float values in `_hyper`
      # could be exchaned between different replicas. By default,
      # if the value is not exchanged the default value is fed. No need
      # to take care of feeding values that are not being exchanged.
      opt = tf.keras.optimizers.get(optimizer_config)
      for n, v in opt._hyper.items():
        if isinstance(v, float):
          opt._set_hyper(n, hp.get_hparam(n, default_value=v))

      train_attrs[i].update({
          'model': model,
          'loss_functions': loss_functions,
          'optimizer': opt
      })
    self._train_attrs = train_attrs
    self.inputs = list(itertools.chain(
        *[train_attrs[i]['model'].inputs for i in range(n_replicas)]))

    self._is_compiled = True

  def _get_metric_tensors(self, metric_name):
    return [self._train_attrs[i][metric_name] for i in range(self.n_replicas)]

  def _get_train_ops(self):
    return [self._train_attrs[i]['train_op'] for i in range(self.n_replicas)]

  @property
  def metrics_names(self):
    names = ['loss_%d' %i for i in range(self.n_replicas)]
    # add more metrics here
    return names

  def test_on_batch(self, x, y):
    """Test all replicas on a single batch of samples."""
    if not tf.executing_eagerly():
      feed_dict = {input_: x for input_ in self.inputs}
      feed_dict.update({
          self._target_tensor: y
      })
      hp_tensors_and_values = (
          self._hp_state_space.prepare_feed_tensors_and_values(training=False))
      feed_dict.update(hp_tensors_and_values)

      metric_tensors = self._get_metric_tensors('loss')

      evaluated = self._run(metric_tensors, feed_dict=feed_dict)
      metrics = evaluated

      return metrics
    else:
      raise NotImplementedError()

  def train_on_batch(self, x, y):
    """Runs a single gradient update on a single batch of data."""
    if not tf.executing_eagerly():
      feed_dict = {input_: x for input_ in self.inputs}
      feed_dict.update({
          self._target_tensor: y
      })
      hp_tensors_and_values = (
          self._hp_state_space.prepare_feed_tensors_and_values(training=True))
      feed_dict.update(hp_tensors_and_values)

      metric_tensors = self._get_metric_tensors('loss')
      ops = self._get_train_ops()
      evaluated = self._run(metric_tensors + ops, feed_dict=feed_dict)
      metrics = evaluated[:len(metric_tensors)]

      return metrics
    else:
      raise NotImplementedError()

  def _run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    sess = tf.compat.v1.keras.backend.get_session()
    return sess.run(fetches,
                    feed_dict=feed_dict,
                    options=options,
                    run_metadata=run_metadata)

  def fit(self,
          x,
          y,
          exchange_hparams,
          validation_split=0.0,
          validation_data=None,
          batch_size=2,
          epochs=1,
          callbacks=None):
    if not self._is_compiled:
      raise ValueError("model is not compiled. Call compile() method first.")
    self._hp_state_space = HPSpaceState(self, exchange_hparams)

    # Create tensors for true labels.
    # A single tensor is fed to all ensemble losses.
    target_tensor = train_utils.create_training_target(
        train_utils.infer_shape_from_numpy_array(y))
    self._target_tensor = target_tensor

    # create losses and optimization step operation
    for i in range(self.n_replicas):
      y_pred = self._train_attrs[i]['model'].outputs[0]
      loss_function = self._train_attrs[i]['loss_functions'][0]
      loss = loss_function(target_tensor, y_pred)
      self._train_attrs[i]['loss'] = loss
      var_list = self._train_attrs[i]['model'].trainable_variables
      train_op = self._train_attrs[i]['optimizer'].get_updates(
          loss, var_list)

      self._train_attrs[i]['train_op'] = train_op

    return model_iteration(self,
                           x,
                           y,
                           exchange_hparams,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks)

  def reset_metrics(self):
    pass

def model_iteration(model,
                    x,
                    y,
                    exchange_hparams,
                    callbacks=None,
                    validation_split=0.0,
                    validation_data=None,
                    batch_size=2,
                    epochs=1,
                    random_data_split_state=0,
                    mode=ModeKeys.TRAIN,
                    verbose=1):
  """
  """
  # TODO: docstring
  # prepare and validate data (TODO: validate)
  if datasets is None:

    datasets = train_utils.prepare_data_iterables(
        x, y, validation_split, validation_data, batch_size=32,
        shuffle=True, shuffle_buf_size=1024,
        random_state=random_data_split_state)

  if len(datasets) == 1:
    do_validation = False
    train_dataset = datasets[0]
  else:
    do_validation = True
    train_dataset, test_dataset = datasets

  # TODO: calculate `steps_per_epochs`

  # TODO: `_print_train_info()` here

  # TODO: deal with use_steps, num_samples, steps_per_epochs,
  # num_samples_or_steps
  # TODO: Add predict aggregator
  steps_per_epoch = x.shape[0] // batch_size
  num_samples_or_steps = x.shape[0]
  aggregator = keras_train_utils.MetricsAggregator(
      use_steps=False,
      num_samples=None if steps_per_epoch else num_samples_or_steps,
      steps=steps_per_epoch)

  # Configure callbacks
  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      samples=num_samples_or_steps,
      epochs=epochs,
      verbose=verbose,
      mode=mode)
  callbacks._call_begin_hook(mode)


  for epoch in range(epochs):
    epoch_logs = {}
    if mode != ModeKeys.PREDICT:
      # Collecting and resetting metrics has non-zero cost and will needlessly
      # slow down model.predict.
      model.reset_metrics()

    if mode == ModeKeys.TRAIN:
      callbacks.on_epoch_begin(epoch, epoch_logs)

    # batch_start and batch_end are added so we can use the
    # Keras' aggregator (which accepts it as args)
    batch_start = 0
    batch_end = batch_size
    for batch_index, (x, y) in enumerate(train_dataset):
      # Callbacks batch_begin.
      batch_logs = {'batch': batch_index, 'size': steps_per_epoch}
      callbacks._call_batch_hook(mode, 'begin', batch_index, batch_logs)

      # Get outputs.
      batch_outs = model.train_on_batch(x, y)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]

      # Aggregate results.
      if batch_index == 0:
        aggregator.create(batch_outs)
      aggregator.aggregate(batch_outs, batch_start, batch_end)

      # Callbacks batch end.
      batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
      callbacks._call_batch_hook(mode, 'end', batch_index, batch_logs)

      # Note: following should be removed in tensorflow versions older
      # than '1.15.2' (to be honest I need to check from which versions
      # to remove below lines). It calls explicitly the `on_batch_end()` of the
      # progbar callback.
      progbar = callbacks.callbacks[2]
      if isinstance(progbar, tf.keras.callbacks.ProgbarLogger):
        progbar.on_batch_end(batch_index, batch_logs)

      batch_start += y.shape[0]
      batch_end += y.shape[0]

    aggregator.finalize()
    results = aggregator.results
    epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
    if len(results) == 1:
      results = results[0]

    if mode == ModeKeys.TRAIN:
      # Epochs only apply to `fit`.
      callbacks.on_epoch_end(epoch, epoch_logs)

  callbacks._call_end_hook(mode)
  callbacks.on_train_end()
  if mode == ModeKeys.TRAIN:
    return model.history
  return results