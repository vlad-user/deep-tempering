import os
from collections import abc
from collections import OrderedDict
import inspect
import json

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle

LOGS_PATH = os.path.join(os.getcwd(), '.deep_tempering_logs')


class HyperParamState:
  def __init__(self, default_values=None):
    self._attrs = {}
    self.default_values = default_values
  def get_hparam(self, name, default_value=None):
    # during creating of the optimal model default (such
    # as dropout etc) could be passed.
    if (isinstance(self.default_values, abc.Mapping)
        and name in self.default_values):
      return self.default_values[name]

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

class HyperParamSpace:
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
    hps = HyperParamSpace(hparams_dict)
    hps.hpspace
    # {0: {'learning_rate': 0.001, 'dropout_rate': 0.0},
    #  1: {'learning_rate': 0.0055000000000000005, 'dropout_rate': 0.3},
    #  2: {'learning_rate': 0.01, 'dropout_rate': 0.6}}
    ```jjj
    """
    self.ensemble_model = ensemble_model
    hparams_dict = dict((k, list(v)) for k, v in hparams_dict.items())
    n_replicas = len(hparams_dict[hparams_dict.__iter__().__next__()])
    self._hyperparameter_names = sorted(list(hparams_dict.keys()))
    self.n_replicas = n_replicas
    self.hpspace = {
        i: {k: v[i] for k, v in hparams_dict.items()}
        for i in range(n_replicas)
    }

  @property
  def hyperparameters_names(self):
    return self._hyperparameter_names

  def swap_between(self, replica_i, replica_j, hyperparam_name):
    """Swaps `hyperparam_name` between `replica_i` and `replica_j`."""
    hp_i = self.hpspace[replica_i][hyperparam_name]
    hp_j = self.hpspace[replica_j][hyperparam_name]
    self.hpspace[replica_j][hyperparam_name] = hp_i
    self.hpspace[replica_i][hyperparam_name] = hp_j

  def get_ordered_hparams(self, name):
    """Returns list of tuples of adjacent `(replica_id, hp_value)`."""
    hparams = [(i, self.hpspace[i][name]) for i in range(self.n_replicas)]
    hparams.sort(key=lambda x: x[1])
    return hparams

  def prepare_feed_tensors_and_values(self, training=True):
    # TODO: replace `training` with ModeKeys instance check

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
        if not training and 'dropout_rate' in hpname:
          value = 0.
        else:
          value = current_hp_dict[hpname][i]
        placeholder = hpstates[i]._get_hparam(hpname)

        assert placeholder is not None
        feed_dict[placeholder] = value

    return feed_dict

def call_metric_function(metric_fn,
                         y_true,
                         y_pred=None,
                         weights=None,
                         mask=None):
  return keras_train_utils.call_metric_function(metric_fn,
                                                y_true,
                                                y_pred,
                                                weights,
                                                mask)

def _infer_init_args_kwargs(cls_):
  """Attempts to extract args/kwargs that class `cls_` has been init'ed with."""
  extracted_signature = OrderedDict()
  signature = inspect.signature(cls_.__init__)
  args_dict = OrderedDict(
      [(k, v.default) for k, v in signature.parameters.items()])
  for arg in args_dict:
    default_value = None if args_dict[arg] == inspect._empty else args_dict[arg]
    extracted_value = None
    try:
      extracted_value = cls_.__dict__[arg]
    except KeyError:
      try:
        extracted_value = cls_.__dict__['_' + arg]
      except KeyError:
        pass
    extracted_signature[arg] = extracted_value or default_value
  return extracted_signature

def infer_shape_from_numpy_array(ary):
  if len(ary.shape) == 1:
    return (None,)
  return (None,) + ary.shape[1:]

def create_training_target(shape, dtype=None):
  dtype = dtype or tf.int32

  if shape[0] is None:
    shape = shape[1:]

  return tf.keras.layers.Input(shape, dtype=dtype)

class MetricsAggregator(keras_train_utils.Aggregator):
  """Aggregator that calculates loss and metrics info.
  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size*num_batches`.
    steps: Total number of steps, ie number of times to iterate over a dataset
      to cover all samples.
  """

  def __init__(self, n_replicas, num_samples=None, steps=None):
    super(MetricsAggregator, self).__init__(
        use_steps=False,
        num_samples=num_samples,
        steps=steps,
        batch_size=None)
    self.n_replicas = n_replicas

  def create(self, batch_outs):
    self.results = [0.] * len(batch_outs)

  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    # Losses.
    for i in range(self.n_replicas):
      self.results[i] += batch_outs[i] * (batch_end - batch_start)

    # Metrics (always stateful, just grab current values.)
    self.results[self.n_replicas:] = batch_outs[self.n_replicas:]


  def finalize(self):
    if not self.results:
      raise ValueError('Empty training data.')

    for i in range(self.n_replicas):
      self.results[i] /= self.num_samples


def prepare_data_iterables(x,
                           y=None,
                           validation_split=0.0,
                           validation_data=None,
                           exchange_split=0.0,
                           exchange_data=None,
                           batch_size=32,
                           epochs=1,
                           shuffle=True,
                           random_state=0):
  
  # during testing DataIterable is passed
  if isinstance(x, DataIterable):
    return x

  # predict mode
  if y is None:
    return DataIterable(x, y, batch_size, epochs, shuffle)

  train_data, validation_data, exchange_data = (
      _train_validation_exchange_data((x, y), validation_data=validation_data,
          exchange_data=exchange_data, validation_split=validation_split,
          exchange_split=exchange_split))

  train_iterable = DataIterable(train_data[0], train_data[1],
                                batch_size=batch_size, epochs=epochs,
                                shuffle=shuffle)
  if validation_data is not None:
    validation_iterable = DataIterable(validation_data[0], validation_data[1],
                                       batch_size=batch_size, epochs=epochs,
                                       shuffle=shuffle)
  else:
    validation_iterable = None

  # exchange data is passed as is

  return (train_iterable, validation_iterable, exchange_data)


  # elif  0.0 < validation_split < 1:
  #   x_train, x_test, y_train, y_test = train_test_split(x, y,
  #       test_size=validation_split, random_state=random_state)
  #   train_dataset = DataIterable(x_train, y_train, batch_size, epochs, shuffle)
  #   test_dataset = DataIterable(x_test, y_test, batch_size, epochs, shuffle)
  #   return [train_dataset, test_dataset]
  # else:
  #   raise ValueError('Cannot parition data.')

def _validate_dataset_shapes(*args):
  shapes = []
  for arg in args:
    if isinstance(arg, (list, tuple)):
      for item in arg:
        shapes.append(item.shape)
    else:
      shapes.append(arg.shape)

  if len(set([s[0] for s in shapes])) != 1:
    raise ValueError('First dimension of inputs and targets must be equal')

class DataIterable:
  """Batch-wise iterable for numpy data."""
  def __init__(self, x, y=None, batch_size=32, epochs=1, shuffle=False):
    x = np.asarray(x)

    if y is not None:
      y = np.asarray(y)
      assert x.shape[0] == y.shape[0]
    self.data = {
        'x': x,
        'y': y
    }
    self.batch_size = batch_size
    self.epochs = epochs
    self.shuffle = shuffle

  def __iter__(self):
    return _NumpyIterator(self.data,
                          self.batch_size,
                          self.epochs,
                          self.shuffle)

  def __len__(self):
    return self.data['x'].shape[0]

class _NumpyIterator:
  def __init__(self, data_dict, batch_size, epochs, shuffle):
    self.data_dict = data_dict
    self.batch_size = batch_size or 128
    self.epochs = epochs
    self.shuffle = shuffle
    self.begin = 0
    self.end = min(self.batch_size, data_dict['x'].shape[0])
    self.epoch_num = 0

  def __next__(self):

    y = self.data_dict['y']
    if self.begin == 0 and self.shuffle and y is not None:
      self.data_dict = arrays_datadict_shuffle(self.data_dict)
      y = self.data_dict['y']

    x = self.data_dict['x']
    if self.begin >= x.shape[0]:
      self.epoch_num += 1
      if self.epoch_num >= self.epochs:
        raise StopIteration()
      else:
        self.begin = 0
        self.end = min(self.batch_size, x.shape[0])

    batch_x = x[self.begin: self.end]
    if y is not None:
      batch_y = y[self.begin: self.end]

    self.begin = self.end
    self.end += self.batch_size

    if y is not None:
      return batch_x, batch_y
    else:
      return batch_x

class _GraphModeIterator:
  def __init__(self, next_elem):
    self.next_elem = next_elem

  def __next__(self):
    try:
      sess = tf.compat.v1.keras.backend.get_session()
      evaled = sess.run(self.next_elem)
    except tf.compat.v1.errors.OutOfRangeError:
      raise StopIteration()
    return evaled['x'], evaled['y']

class GraphModeDataIterable:
  # TODO: Extend support for eager iteration
  # Implement Wrapper for other than numpy arrays data types.

  def __init__(self, x, y, batch_size=32, epochs=1, shuffle=True, shuffle_buf_size=1024):

    self.batch_size = min(batch_size or 128, y.shape[0])
    self.epochs = epochs
    self.shuffle = shuffle
    self.shuffle_buf_size = shuffle_buf_size
    self.__len = x.shape[0]
    _validate_dataset_shapes(*[x, y])
    d = tf.data.Dataset.from_tensor_slices({
        'x': x,
        'y': y
    })
    d = d.repeat(self.epochs)
    if self.shuffle:
      d = d.shuffle(self.shuffle_buf_size)
    d = d.batch(self.batch_size)

    self._iterator = d.make_initializable_iterator()
    self._next = self._iterator.get_next()

  def __iter__(self):
    # When invoking validation in training loop, avoid creating iterator and
    # list of feed values for the same validation dataset multiple times (which
    # essentially would call `iterator.get_next()` that slows down execution
    # and leads to OOM errors eventually.
    if not tf.executing_eagerly():
      sess = tf.compat.v1.keras.backend.get_session()
      sess.run(self._iterator.initializer)
      return _GraphModeIterator(self._next)
    else:
      raise NotImplementedError()

  def __len__(self):
    return self.__len

def arrays_datadict_shuffle(datadict):
  """Shuffles all values in `dict` in unison."""
  indices = np.arange(datadict['x'].shape[0])
  np.random.shuffle(indices)

  return {
      k: np.take(v, indices=indices, axis=0) if v is not None else v
      for k, v in datadict.items()
  }

def _train_validation_exchange_data(train_data,
                                    validation_data=None,
                                    exchange_data=None,
                                    validation_split=0.0,
                                    exchange_split=0.0,
                                    random_state=0,
                                    verbose=0):
  """Extracts exchange data.
  Extraction of the exchanges dataset is done in the following order:

  * If exchange data is passes explicitly, then use this data.
  * Else If `exchange_split` is non-zero sample randomly out of train
  * Else If validation data is passed explicitly, then use this data.
    data.
  * Else raise `ValueError
  """
  # validate train data

  assert all(isinstance(a, np.ndarray) for a in train_data)

  x_train, y_train = train_data

  # extract validation data
  if validation_data is not None:
    validation_data = validation_data
  elif validation_split > 0.0:
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, random_state=0, test_size=validation_split)
    validation_data = (x_validation, y_validation)

  # extract exchange data
  if exchange_data is not None:
    # exchange data is passed explicitly
    exchange_data = exchange_data
  elif exchange_split > 0:
    # exchange data is not passed explicitly, but `exchange_split > 0`
    x_train, x_exchange, y_train, y_exchange = train_test_split(
        x_train, y_train, random_state=random_state, test_size=exchange_split)
    # exchange data is not passed and exchange_split == 0, using validation_data
    exchange_data = (x_exchange, y_exchange)
  elif validation_data is not None:
    exchange_data = validation_data

  # Maybe show as warning?
  # if exchange_data is None:
  #   err_msg = ("Unable to extract exchange dataset. Pass exchange dataset "
  #              "explicitly as argument to `fit()`, or specify "
  #              "`exchange_split` argument to greater than 0, or "
  #              "pass validation_data or `validation_split > 0`.")
  #   raise ValueError(err_msg)

  return (x_train, y_train), validation_data, exchange_data

def gpu_device_name(replica_id):
  """Returns a device name on which is replica is going to executed."""
  if tf.__version__ < '2.1.0':
    gpus = tf.config.experimental.list_physical_devices('GPU')
  else:
    gpus = tf.config.list_physical_devices('GPU')

  gpus_names = [g.name for g in gpus]
  if not gpus_names:
    return '/cpu:0'

  return gpus_names[replica_id % len(gpus_names)]

def min_or_max_for_metric(metric_name):
  """Decides whether optimal value for `metric_name` is being maximized or minimized"""
  min_metrics = ['loss', 'error']
  max_metrics = ['acc', 'accuracy', 'precision', 'recall', 'auc']

  if any(m in metric_name for m in min_metrics):
    return 'min'
  elif any(m in metric_name for m in max_metrics):
    return 'max'
  else:
    return 'min'

def load_optimal_model(model_builder, hyperparams=None, path=None):
  """Loads optimal model stored by instance of `MonitorOptimalModelCallback`.

  Args:
    model_builder: A function that creates a not compiled keras' model.
    hyperparams: (Optional) A dictionary of hyperparameters (such as dropout)
      you wish to instantiate your model with. If nothing is specified the
      model is created with hyperparameters of the optimal model at that time.
    path: (Optional) A path where the model is stored (in case the model's
      weights were not stored at the default place).
  """
  path = path or LOGS_PATH
  model_weights_path = os.path.join(path, 'optimal_model.h5')
  if hyperparams is None:
    with open(os.path.join(path, 'hyperparams.json')) as fo:
      hyperparams = json.load(fo)
  hp = HyperParamState(default_values=hyperparams)
  model = model_builder(hp)
  model.load_weights(model_weights_path)
  return model

  

