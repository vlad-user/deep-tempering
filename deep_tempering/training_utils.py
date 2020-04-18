from collections import abc
from collections import OrderedDict
import inspect

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as arrays_shuffle

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
                           batch_size=32,
                           epochs=1,
                           shuffle=True,
                           random_state=0):

  if isinstance(x, DataIterable):
    return [x]

  if validation_split == 0.0 and validation_data is None:
    return [DataIterable(x, y, batch_size, epochs, shuffle)]

  elif validation_split == 0.0 and validation_data is not None:
    train_dataset = DataIterable(x, y, batch_size, epochs, shuffle)
    test_dataset = DataIterable(validation_data[0],
                                validation_data[1],
                                batch_size,
                                epochs,
                                shuffle)
    return [train_dataset, test_dataset]

  elif  0.0 < validation_split < 1:
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=validation_split, random_state=random_state)
    train_dataset = DataIterable(x_train, y_train, batch_size, epochs, shuffle)
    test_dataset = DataIterable(x_test, y_test, batch_size, epochs, shuffle)
    return [train_dataset, test_dataset]
  else:
    raise ValueError('Cannot parition data.')

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
    x = self.data_dict['x']
    y = self.data_dict['y']
    if self.begin == 0 and self.shuffle and y is not None:
      x, y = arrays_shuffle(x, y)
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