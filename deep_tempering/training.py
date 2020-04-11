import itertools
from collections import abc

import tensorflow as tf
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np
import tqdm

from deep_tempering import training_utils
from deep_tempering import callbacks as cbks

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
              metrics=None,
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
          shuffle=True,
          validation_freq=1,
          verbose=1,
          callbacks=None):
    if not self._is_compiled:
      raise ValueError("model is not compiled. Call compile() method first.")
    self._hp_state_space = HPSpaceState(self, exchange_hparams)

    # Create tensors for true labels.
    # A single tensor is fed to all ensemble losses.
    target_tensor = training_utils.create_training_target(
        training_utils.infer_shape_from_numpy_array(y))
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
                           validation_split=validation_split,
                           validation_data=validation_data,
                           validation_freq=validation_freq,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           shuffle=shuffle,
                           verbose=verbose)

  def reset_metrics(self):
    pass

def _make_execution_function(model, mode):
  if mode == ModeKeys.TRAIN:
    return model.train_on_batch
  elif mode == ModeKeys.TEST:
    return model.test_on_batch
  elif mode == ModeKeys.PREDICT:
    raise NotImplementedError()
  else:
    raise ValueError('Unrecognized mode: ', mode)

def model_iteration(model,
                    inputs,
                    targets=None,
                    batch_size=2,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    shuffle=False,
                    steps_per_epoch=None,
                    validation_split=0.0,
                    validation_freq=1,
                    random_data_split_state=0,
                    mode=ModeKeys.TRAIN):
  """Loop function for arrays of data with modes TRAIN/TEST/PREDICT.
  Args:
    model: `EnsembleModel` instance.
    inputs: Either a list or dictionary of arrays, or a dataset instance.
    targets: List/dictionary of input arrays.
    batch_size: Integer batch size or None if unknown.
    epochs: Number of times to iterate over the data
    verbose: 0, 1, or 2. Verbosity mode.
      0 = silent, 1 = progress bar, 2 = one line per epoch.
      Note that the progress bar is not particularly useful when
      logged to a file, so verbose=2 is recommended when not running
      interactively (eg, in a production environment).
    callbacks: List of callbacks to be called during training
    validation_data:
    shuffle: Whether to shuffle the data at the beginning of each epoch
      concatenation of list the display names of the outputs of `f` and the
      list of display names of the outputs of `f_val`.
    steps_per_epoch: Total number of steps (batches of samples) before
      declaring one epoch finished and starting the next epoch. Ignored with
      the default value of `None`.
    validation_freq: Only relevant if validation data is provided. Integer or
      `collections_abc.Container` instance (e.g. list, tuple, etc.). If an
      integer, specifies how many training epochs to run before a new
      validation run is performed, e.g. `validation_freq=2` runs
      validation every 2 epochs. If a Container, specifies the epochs on
      which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
      validation at the end of the 1st, 2nd, and 10th epochs.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.


  Returns:
    - In TRAIN mode: `History` object.
    - In TEST mode: Evaluation metrics.
    - In PREDICT mode: Outputs of the Model called on inputs.
  Raises:
    ValueError: in case of invalid arguments.
  """
  # TODO: docstring
  # prepare and validate data (TODO: validate)

  datasets = training_utils.prepare_data_iterables(
      inputs, targets, validation_split, validation_data, batch_size=batch_size,
      shuffle=shuffle, shuffle_buf_size=1024,
      random_state=random_data_split_state)

  if len(datasets) == 1:
    do_validation = False
    train_dataset = datasets[0]
  else:
    do_validation = True
    train_dataset, test_dataset = datasets
    validation_steps = len(test_dataset)# // batch_size

  # TODO: calculate `steps_per_epochs`

  f = _make_execution_function(model, mode)

  # TODO: `_print_train_info()` here

  # TODO: deal with use_steps, num_samples, steps_per_epochs,
  # num_samples_or_steps

  steps_per_epoch = steps_per_epoch or len(train_dataset)# // batch_size
  num_samples_or_steps = len(train_dataset)

  # TODO: Add predict aggregator
  aggregator = training_utils.MetricsAggregator(
      n_replicas=model.n_replicas,
      num_samples=len(train_dataset))

  # Configure callbacks
  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      samples=num_samples_or_steps,
      epochs=epochs,
      verbose=0,
      mode=mode)

  # Since tensorflow 2.2 the progbar callback could be taken care of
  # within the `CallbackList` instance. Currently, it is implemented
  # separately. See the following commit for more:
  # https://github.com/tensorflow/tensorflow/commit/10666c59dd4858645d1b03ce01f4450da80710ec
  progbar = cbks.get_progbar(model)
  progbar.params = callbacks.params
  progbar.params['verbose'] = verbose

  callbacks.model.stop_training = False # maybe remove this
  callbacks._call_begin_hook(mode)
  progbar.on_train_begin()


  for epoch in range(epochs):
    if callbacks.model.stop_training:
      break

    # Setup work for each epoch
    epoch_logs = {}
    if mode != ModeKeys.PREDICT:
      # Collecting and resetting metrics has non-zero cost and will needlessly
      # slow down model.predict.
      model.reset_metrics()

    if mode == ModeKeys.TRAIN:
      callbacks.on_epoch_begin(epoch, epoch_logs)
    progbar.on_epoch_begin(epoch, epoch_logs)

    # batch_start and batch_end are added so we can use the
    # Keras' aggregator. It accepts it as args to compute
    # weighted batch size average of the overall losses.
    batch_start = 0
    for batch_index, (x, y) in enumerate(train_dataset):
      # Callbacks batch_begin.
      batch_end = batch_start + y.shape[0]
      batch_logs = {'batch': batch_index, 'size': y.shape[0]}
      callbacks._call_batch_hook(mode, 'begin', batch_index, batch_logs)
      progbar.on_batch_begin(batch_index, batch_logs)

      # Get outputs.
      batch_outs = f(x, y)

      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]

      # Aggregate results.
      if batch_index == 0:
        aggregator.create(batch_outs)
      aggregator.aggregate(batch_outs, batch_start, batch_end)

      # Callbacks batch end.
      batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
      callbacks._call_batch_hook(mode, 'end', batch_index, batch_logs)
      progbar.on_batch_end(batch_index, batch_logs)

      batch_start = batch_end

      if callbacks.model.stop_training:
        break

    aggregator.finalize()
    results = aggregator.results
    epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
    if len(results) == 1:
      results = results[0]

    if (do_validation and
        keras_train_utils.should_run_validation(validation_freq, epochs)):
      val_results = model_iteration(
          model,
          test_dataset,
          targets=None,
          batch_size=batch_size,
          steps_per_epoch=validation_steps,
          callbacks=callbacks,
          verbose=0,
          mode=ModeKeys.TEST)
      if not isinstance(val_results, list):
        val_results = [val_results]

      epoch_logs = cbks.make_logs(
          model, epoch_logs, val_results, mode, prefix='val_')

    if mode == ModeKeys.TRAIN:
      # Epochs only apply to `fit`.
      callbacks.on_epoch_end(epoch, epoch_logs)
    progbar.on_epoch_end(epoch, epoch_logs)

  callbacks._call_end_hook(mode)
  callbacks.on_train_end()
  if mode == ModeKeys.TRAIN:
    return model.history
  return results