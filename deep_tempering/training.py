import itertools
import inspect
import copy
from collections import abc
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import numpy as np
import tqdm



from deep_tempering import training_utils
from deep_tempering import callbacks as cbks



class EnsembleModel:
  """Mimics the behaviour of `keras.Model` for ensemble PT training."""
  def __init__(self, model_builder):
    """Instantiates a new PTEnsemble instance."""
    if not callable(model_builder):
      raise TypeError("Expected callable `model_builder`.")

    self._model_builder_fn = model_builder
    self._is_compiled = False
    self._built_losses_metrics_optimizer = False
    self.run_eagerly = False
    self._compile_metrics = None
    self._stateful_metrics_names = None

    # Stores attributes such as losses, optimizers, states of hyperparameters
    # as dictionary {replica_id: {loss:..., optimizer:..,}}
    self._train_attrs = None

    # HyperParamStateSpace instance could be instantiated only when exchange
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
    metrics = metrics or []
    self._stateful_metrics_names = _stateful_metrics_names(metrics)

    # set config for `tf.keras.optimizers.Optimizer` subclasses
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
      config = optimizer.get_config()
      optimizer_config = {
          'class_name': config['name'],
          'config': config
    }
    elif isinstance(optimizer, str):
      # TODO: Add test for string optimizer. Check if config == {}
      # then `_set_hyper(placeholder)` works as expected.
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
    optimizer_config['config'].pop('name', None)

    train_attrs = {i: {} for i in range(self.n_replicas)}
    for i in range(self.n_replicas):
      with tf.device(training_utils.gpu_device_name(i)):
        # build model and the state of hyperparameters
        with tf.variable_scope('model_%d' % i):
          hp = training_utils.HyperParamState()
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

        # Each replica will have now its own metric class. 
        # For example, if `tf.keras.metrics.Precision()`
        # is passed to metrics we will duplicate this class
        # `n_replica` times.
        # TODO: right now if the initialized class is passed it is not
        # used. A new class is created instead. This could be improved
        # by using this class for the first replica. Same could be done
        # of initialized optimizer.
        compiled_metrics = []
        for m in metrics:
          if isinstance(m, str):
            m = m
          elif isinstance(m, tf.keras.metrics.Metric):
            args_kwargs = training_utils._infer_init_args_kwargs(m)
            _ = args_kwargs.pop('name', None)
            m = m.__class__(**args_kwargs)
          elif callable(m):
            m = m
          else:
            raise ValueError('unexpected metric', m)
          compiled_metrics.append(m)

      train_attrs[i].update({
          'model': model,
          'loss_functions': loss_functions,
          'optimizer': opt,
          'compiled_metrics': compiled_metrics
      })

    self._train_attrs = train_attrs
    self.inputs = list(itertools.chain(
        *[train_attrs[i]['model'].inputs for i in range(n_replicas)]))
    self.outputs = list(itertools.chain(
        *[train_attrs[i]['model'].outputs for i in range(n_replicas)]))
    self._is_compiled = True

  def summary(self, line_length=None, positions=None, print_fn=None):
    if not self._is_compiled:
      raise ValueError('Unable to get summary. The model hasn\'t been built yet.')
    else:
      self._train_attrs[0]['model'].summary(line_length=line_length, positions=positions, print_fn=print_fn)


  @property
  def metrics_names(self):
    # losses
    names = ['loss_%d' %i for i in range(self.n_replicas)]
    # the rest of the metrics
    if self._stateful_metrics_names:
      for m in self._stateful_metrics_names:
        if m == 'accuracy':
          m = 'acc'
        names += [m + '_%d' %i for i in range(self.n_replicas)]

    return names

  @property
  def models(self):
    return [self._train_attrs[i]['model'] for i in range(self.n_replicas)]

  def predict_on_batch(self, x, y):
    if not tf.executing_eagerly():
      feed_dict = {input_: x for input_ in self.inputs}
      hp_tensors_and_values = (
          self.hpspace.prepare_feed_tensors_and_values(training=False))
      feed_dict.update(hp_tensors_and_values)
      evaluated = self._run(self.outputs, feed_dict)

      return evaluated
    else:
      raise NotImplementedError()

  def test_on_batch(self, x, y):
    """Test all replicas on a single batch of samples."""
    if not tf.executing_eagerly():
      feed_dict = {input_: x for input_ in self.inputs}
      feed_dict.update({
          self._target_tensor: y
      })
      hp_tensors_and_values = (
          self.hpspace.prepare_feed_tensors_and_values(training=False))
      feed_dict.update(hp_tensors_and_values)

      metric_tensors = self._get_metric_tensors('loss')
      for metric_name in self._stateful_metrics_names:
        metric_tensors += self._get_metric_tensors(metric_name)

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
          self.hpspace.prepare_feed_tensors_and_values(training=True))
      feed_dict.update(hp_tensors_and_values)

      metric_tensors = self._get_metric_tensors('loss')

      for metric_name in self._stateful_metrics_names:
        metric_tensors += self._get_metric_tensors(metric_name)


      ops = self._get_train_ops()
      evaluated = self._run(metric_tensors + ops, feed_dict=feed_dict)
      metrics = evaluated[:len(metric_tensors)]

      return metrics
    else:
      raise NotImplementedError()

  def optimal_model(self, metric_name='loss'):
    """Returns optimal based on latest metrics log model.

    Args:
      metric_name: The name of the metric (or string that the metric
        name starts with).

    Raises:
      ValueError: If the metric name could not be uniquely identified.

    Returns:
      Not compiled keras model.
    """
    # NOTE: If to remove this function the 
    # training_test.test_metrics_and_losses() must be modified.

    # decide optimal based on argmax or argmin of the metric
    if 'loss' in metric_name or 'error' in metric_name:  #ToDo: change to method param
      argfn = np.argmin
    else:
      argfn = np.argmax

    logs = {k: v[-1] for k, v in self.history.history.items()}
    ordered_metrics = cbks.get_ordered_metrics(logs)
    if len(ordered_metrics) != self.n_replicas:
      raise ValueError('Cannot extract metrics for', metric_name)

    optimal_replica_id = argfn([x[1] for x in ordered_metrics])
    return self._train_attrs[optimal_replica_id]['model']


  def _run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Wraps tensorflow session's `run()` method."""
    # TODO: set first session of soft placement configuration
    sess = tf.compat.v1.keras.backend.get_session()

    return sess.run(fetches,
                    feed_dict=feed_dict,
                    options=options,
                    run_metadata=run_metadata)

  def _build_losses_metrics_optimizer(self, target_ary):
    """Builds the logic that goes after `logits`.

    This function is invoked once we know the target tensor shape
    during invocation of `fit()`, `evaluate()` or `predict()`.

    Args:
      target_ary: Target input data.
    """
    if not self._is_compiled:
      raise ValueError("model is not compiled. Call compile() method first.")

    # Create tensors for true labels.
    # A single tensor is fed to all ensemble losses.
    target_tensor_shape = training_utils.infer_shape_from_numpy_array(target_ary)
    target_tensor = training_utils.create_training_target(target_tensor_shape)
    self._target_tensor = target_tensor

    # create losses and optimization step operation
    for i in range(self.n_replicas):
      with tf.device(training_utils.gpu_device_name(i)):
        model = self._train_attrs[i]['model']
        loss_functions = self._train_attrs[i]['loss_functions']
        compiled_metrics = self._train_attrs[i]['compiled_metrics']

        # The target tensor is ready. Create metric tensors.
        output_names = [o.name for o in model.outputs]
        output_shapes = [o.shape for o in model.outputs]
        per_output_metrics = keras_train_utils.collect_per_output_metric_info(
            compiled_metrics, output_names, output_shapes,
            loss_functions)[0]

        metrics_dict = OrderedDict()
        for (_, metric_wrapper), name in zip(per_output_metrics.items(),
                                             self._stateful_metrics_names):
          metrics_dict[name] = metric_wrapper 

        self._train_attrs[i]['metrics_dict'] = metrics_dict
        metrics_names = [p[0] for p in metrics_dict.items()]
        metrics_tensors = self._handle_metrics(model.outputs,
                                              [self._target_tensor],
                                              metrics_dict)
        self._train_attrs[i]['metrics_tensors_dict'] = OrderedDict(
            [(n, t) for n, t in zip(self._stateful_metrics_names, metrics_tensors)])

        # create losses
        y_pred = model.outputs[0]
        loss_function = loss_functions[0]
        loss = loss_function(target_tensor, y_pred)
        self._train_attrs[i]['loss'] = loss

        # create optimization step op
        var_list = model.trainable_variables
        train_op = self._train_attrs[i]['optimizer'].get_updates(
            loss, var_list)

        self._train_attrs[i]['train_op'] = train_op

    self._built_losses_metrics_optimizer = True

  def fit(self,
          x,
          y,
          hyper_params,
          validation_split=0.0,
          validation_data=None,
          exchange_split=0.0,
          exchange_data=None,
          batch_size=2,
          epochs=1,
          shuffle=True,
          validation_freq=1,
          verbose=1,
          callbacks=None,
          swap_step=None,
          burn_in=None,
          **kwargs):
    
    if self._hp_state_space is None:
      self._hp_state_space = training_utils.ScheduledHyperParamSpace(
          self, hyper_params)

    if len(y.shape) == 1:
      y = y[:, None]


    if not self._built_losses_metrics_optimizer:
      self._build_losses_metrics_optimizer(y)

    return model_iteration(self,
                           x,
                           y,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           exchange_data=exchange_data,
                           exchange_split=exchange_split,
                           validation_freq=validation_freq,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           shuffle=shuffle,
                           verbose=verbose,
                           burn_in=burn_in,
                           swap_step=swap_step,
                           **kwargs)

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               callbacks=None):
    """Returns the loss value & metrics values for the model in test mode.
    Computation is done in batches.

    Args:
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors,
          if the model has named inputs.
        - A `tf.data` dataset.
        - A generator or `keras.utils.Sequence` instance.
      y: Target data. Like the input data `x`,
        it could be either Numpy array(s) or TensorFlow tensor(s).
        It should be consistent with `x` (you cannot have Numpy inputs and
        tensor targets, or inversely).
        If `x` is a dataset, generator or
        `keras.utils.Sequence` instance, `y` should not be specified (since
        targets will be obtained from the iterator/dataset).
      batch_size: Integer or `None`.
          Number of samples per gradient update.
          If unspecified, `batch_size` will default to 32.
          Do not specify the `batch_size` is your data is in the
          form of symbolic tensors, dataset,
          generators, or `keras.utils.Sequence` instances (since they generate
          batches).
      verbose: 0 or 1. Verbosity mode.
          0 = silent, 1 = progress bar.
      callbacks: List of `keras.callbacks.Callback` instances.
          List of callbacks to apply during evaluation.
          See [callbacks](/api_docs/python/tf/keras/callbacks).

    Returns:
      Scalar test loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.
    Raises:
      ValueError: in case of invalid arguments.
    """
    if len(y.shape) == 1:
      y = y[:, None]

    if not self._built_losses_metrics_optimizer:
      self._build_losses_metrics_optimizer(y)

    return model_iteration(self,
                           x,
                           y,
                           batch_size=batch_size,
                           epochs=1,
                           callbacks=callbacks,
                           verbose=verbose,
                           mode=ModeKeys.TEST)

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              callbacks=None):
    """Generates output predictions for the input samples.
    Computation is done in batches.
    Args:
      x: Input samples. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A `tf.data` dataset.
        - A generator or `keras.utils.Sequence` instance.
      batch_size: Integer or `None`.
          Number of samples per gradient update.
          If unspecified, `batch_size` will default to 32.
          Do not specify the `batch_size` is your data is in the
          form of symbolic tensors, dataset,
          generators, or `keras.utils.Sequence` instances (since they generate
          batches).
      verbose: Verbosity mode, 0 or 1.
      callbacks: List of `keras.callbacks.Callback` instances.
          List of callbacks to apply during prediction.
          See [callbacks](/api_docs/python/tf/keras/callbacks).

    Returns:
        Numpy array(s) of predictions.
    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    return model_iteration(self,
                           x,
                           targets=None,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           verbose=verbose,
                           mode=ModeKeys.PREDICT)

  def reset_metrics(self):
    for i in range(self.n_replicas):
      for _, v in self._train_attrs[i]['metrics_dict'].items():
        v.reset_states()

  def _handle_metrics(self,
                      outputs,
                      targets=None,
                      per_outputs_metrics=None):
    """Handles calling metric functions.

    Args:
      outputs: List of outputs (predictions).
      targets: List of targets.

    Returns:
      A list of metric result tensors.
    """
    metric_results = []
    if not tf.executing_eagerly():
      with tf.name_scope('metrics'):
        for i in range(len(outputs)):
          output = outputs[i] if outputs else None
          target = targets[i] if targets else None
          output_mask = None # to be implemented
          metric_results.extend(
              self._handle_per_output_metrics(per_outputs_metrics,
                                              target, output))
      return metric_results

    else:
      raise NotImplementedError()

  def _handle_per_output_metrics(self,
                                 metrics_dict,
                                 y_true,
                                 y_pred):
    """Calls metric functions for a single output.

    Args:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      y_true: Target output.
      y_pred: Predicted output.
      mask: Computed mask value for the current output.
    Returns:
      A list of metric result tensors.
    """
    metric_results = []
    for metric_name, metric_fn in metrics_dict.items():
      with tf.name_scope(metric_name):
        metric_result = training_utils.call_metric_function(
            metric_fn, y_true, y_pred, weights=None, mask=None)
        metric_results.append(metric_result)
    return metric_results


  def _get_metric_tensors(self, metric_name):
    """Metrics/losses tensors for each replica."""
    try:
      # loss
      result = [self._train_attrs[i][metric_name] for i in range(self.n_replicas)]
    except KeyError:
      try:
        result = [self._train_attrs[i]['metrics_tensors_dict'][metric_name]
                  for i in range(self.n_replicas)]

      except KeyError:
        raise ValueError("Tensor", metric_name, "doesn't exist")
      except:
        raise

    return result

  def _get_train_ops(self):
    return [self._train_attrs[i]['train_op'] for i in range(self.n_replicas)]

  @property
  def hpspace(self):
    return self._hp_state_space.get_current_hyperparams_space()


def model_iteration(model,
                    inputs,
                    targets=None,
                    batch_size=2,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    exchange_data=None,
                    shuffle=False,
                    validation_split=0.0,
                    exchange_split=0.0,
                    validation_freq=1,
                    random_data_split_state=0,
                    mode=ModeKeys.TRAIN,
                    swap_step=None,
                    burn_in=None,
                    **kwargs):
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
  datasets = training_utils.prepare_data_iterables(
      inputs, targets, validation_split=validation_split,
      validation_data=validation_data, exchange_data=exchange_data,
      batch_size=batch_size, shuffle=shuffle, exchange_split=exchange_split,
      random_state=random_data_split_state)

  val_samples_or_steps = None
  exchange_data = None
  if (isinstance(datasets, training_utils.DataIterable) or
      len(datasets) == 3 and datasets[1] is None):
    # TEST, PREDICT modes or TRAIN mode without validation data
    do_validation = False
    if isinstance(datasets, training_utils.DataIterable):
      train_dataset = datasets
    else:
      train_dataset = datasets[0]
      exchange_data = datasets[2]
  else:
    # TRAIN mode with validation data
    do_validation = True
    train_dataset, test_dataset, exchange_data = datasets
    val_samples_or_steps = len(test_dataset)

  num_samples_or_steps = len(train_dataset)

  f = _make_execution_function(model, mode) #whether to train, test or predict

  if mode == ModeKeys.PREDICT:
    aggregator = keras_train_utils.OutputsAggregator(
        use_steps=False,
        num_samples=num_samples_or_steps)
  else:
    aggregator = training_utils.MetricsAggregator(
        n_replicas=model.n_replicas,
        num_samples=len(train_dataset))

  if mode == ModeKeys.TRAIN and verbose:
    _print_train_info(num_samples_or_steps,
                      val_samples_or_steps,
                      replicas=model.n_replicas,
                      increment='samples')

  # Configure callbacks
  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      samples=num_samples_or_steps,
      epochs=epochs,
      verbose=verbose,
      mode=mode,
      exchange_data=exchange_data,
      swap_step=swap_step,
      burn_in=burn_in,
      **kwargs
  )

  callbacks.model.stop_training = False
  callbacks._call_begin_hook(mode)


  for epoch in range(epochs):
    if callbacks.model.stop_training:
      break

    # Setup work for each epoch
    epoch_logs = {}
    if mode != ModeKeys.PREDICT:
      # Collecting and resetting metrics has non-zero cost and will needlessly
      # slow down model.predict.
      model.reset_metrics()

    callbacks._call_epoch_hook(mode, 'begin', epoch, epoch_logs)

    # batch_start and batch_end are added so we can use the
    # Keras' aggregator. It accepts it as args to compute
    # weighted batch size average of the overall losses.
    batch_start = 0
    for batch_index, batch_data in enumerate(train_dataset):
      # Callbacks batch_begin.
      if mode == ModeKeys.PREDICT:
        x, y = batch_data, None
      else:
        x, y = batch_data

      batch_end = batch_start + x.shape[0]
      batch_logs = {'batch': batch_index, 'size': x.shape[0]}
      callbacks._call_batch_hook(mode, 'begin', batch_index, batch_logs)

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
          callbacks=callbacks,
          verbose=0,
          mode=ModeKeys.TEST)
      if not isinstance(val_results, list):
        val_results = [val_results]

      epoch_logs = cbks.make_logs(
          model, epoch_logs, val_results, mode, prefix='val_')

    callbacks._call_epoch_hook(mode, 'end', epoch, epoch_logs)

  callbacks._call_end_hook(mode)
  callbacks.on_train_end() # TODO: move this to CallbackListWrap

  if mode == ModeKeys.TRAIN:
    return model.history
  return results

def _print_train_info(num_samples_or_steps, val_samples_or_steps, replicas, increment):
  msg = 'Train on {0} {increment}'.format(
      num_samples_or_steps, increment=increment)
  if val_samples_or_steps:
    msg += ', validate on {0} {increment}'.format(
        val_samples_or_steps, increment=increment)
  msg += '. Ensemble of size {0}.'.format(replicas)
  print(msg)

def _make_execution_function(model, mode):
  if mode == ModeKeys.TRAIN:
    return model.train_on_batch
  elif mode == ModeKeys.TEST:
    return model.test_on_batch
  elif mode == ModeKeys.PREDICT:
    return model.predict_on_batch
  else:
    raise ValueError('Unrecognized mode: ', mode)

def _stateful_metrics_names(metrics):
  """Returns all stateful metrics from in `metrics`."""
  stateful_metrics = []
  for m in metrics:
    if m in ['accuracy', 'acc']:
      name = 'acc'
        
    elif isinstance(m, tf.keras.metrics.Metric):
      name = m.name
    elif callable(m):
      name = m.__name__
    else:
      raise ValueError('unrecognized_metric')

    stateful_metrics.append(name)

  return stateful_metrics


