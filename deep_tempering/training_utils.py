# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import collections
from collections import abc as c_abc
from collections import OrderedDict
import multiprocessing.pool
import inspect
import json
import abc
import atexit
import threading
import time

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc

import numpy as np
import six
from six.moves import zip
from sklearn.model_selection import train_test_split

from deep_tempering import composite_tensor_utils


LOGS_PATH = os.path.join(os.getcwd(), '.deep_tempering_logs')

with tf.compat.v1.keras.backend.get_session().graph.as_default():
  _IS_TRAINING_PLACEHOLDER = tf.compat.v1.placeholder_with_default(True, shape=())

def get_training_phase_placeholder():
  return _IS_TRAINING_PLACEHOLDER

class HyperParamState:
  def __init__(self, default_values=None):
    self._attrs = {}
    self.default_values = default_values
  def get_hparam(self, name, default_value=None):
    # during creating of the optimal model default (such
    # as dropout etc) could be passed.
    if (isinstance(self.default_values, c_abc.Mapping)
        and name in self.default_values):
      return self.default_values[name]

    if name in self._attrs:
      raise ValueError('Hyper Params with name ', name, 'already exists.')

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

class ScheduledHyperParamSpace:
  def __init__(self, ensemble_model, hparams_dict):
    self.ensemble_model = ensemble_model
    if all(isinstance(s, str) for s in hparams_dict):
      # not scheduled hyperparams, start scheduling at step 1
      hparams_dict = {1: hparams_dict}

    self.hparams_spaces = {
        step: HyperParamSpace(ensemble_model, hparams_dict[step])
        for step in hparams_dict
    }

  def get_current_hyperparams_space(self):
    curr_step = self._get_current_scheduled_step()
    return self.hparams_spaces[curr_step]

  def _get_current_scheduled_step(self):
    """Returns starting step for which current hyperparams are valid."""
    scheduled_steps = sorted(list(self.hparams_spaces.keys()))
    glob_step = getattr(
        self.ensemble_model, 'global_step', min(scheduled_steps))
    curr_step = scheduled_steps[0]

    for i in range(1, len(scheduled_steps)):
      if glob_step >= scheduled_steps[i]:
        curr_step = scheduled_steps[i]

    return curr_step


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
    ```
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

  def set_hyperparams(self, value, hp_name, replica_id):
    self.hpspace[replica_id][hp_name] = value

  @property
  def hyperparameters_names(self):
    return self._hyperparameter_names

  def swap_between(self, replica_i, replica_j, hyperparam_name):
    """Swaps `hyperparam_name` between `replica_i` and `replica_j`."""
    hp_i = self.hpspace[replica_i][hyperparam_name]
    hp_j = self.hpspace[replica_j][hyperparam_name]
    self.hpspace[replica_j][hyperparam_name] = hp_i
    self.hpspace[replica_i][hyperparam_name] = hp_j

  def copy_hyperparams(self, src_replica, dst_replica, hyperparam_names=None):
    """Copies hyperparams in hyperparam_names from src to dst replica."""
    hyperparam_names = hyperparam_names or self.hyperparameters_names

    for hpname in hyperparam_names:
      self.hpspace[dst_replica][hpname] = self.hpspace[src_replica][hpname]

  def perturb_hyperparams(self, replica_id, hyperparam_name, dist_fn):
    """Perturbs `hyperparam_name` of `replica_id` by the value generated by `dist_fn()`."""
    self.hpspace[replica_id][hyperparam_name] += dist_fn()
    # self.hpspace[replica_id][hyperparam_name] *= dist_fn()

  def get_ordered_hparams(self, name):
    """Returns list of tuples of adjacent `(replica_id, hp_value)`."""
    hparams = [(i, self.hpspace[i][name]) for i in range(self.n_replicas)]
    hparams.sort(key=lambda x: x[1])
    return hparams



  def prepare_feed_tensors_and_values(self, training=True):
    # TODO: replace `training` with ModeKeys instance check

    assert training in {True, False}

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

    feed_dict[get_training_phase_placeholder()] = training

    return feed_dict

# def call_metric_function(metric_fn,
#                          y_true,
#                          y_pred=None,
#                          weights=None,
#                          mask=None):
#   return call_metric_function(metric_fn,
#                               y_true,
#                               y_pred,
#                               weights,
#                               mask)

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

_COPY_THREADS = 4
_COPY_POOL = None


def get_copy_pool():
  """Shared threadpool for copying arrays.

  Pool instantiation takes ~ 2ms, so a singleton pool is used rather than
  creating a pool per SliceAggregator.

  Returns:
    The global copy threadpool.
  """
  global _COPY_POOL
  if _COPY_POOL is None:
    _COPY_POOL = multiprocessing.pool.ThreadPool(_COPY_THREADS)
    atexit.register(_COPY_POOL.close)
  return _COPY_POOL

@six.add_metaclass(abc.ABCMeta)
class Aggregator(object):
  """Abstract base class used to aggregate batch-level outputs of a loop.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size * num_batches`.
    steps: Total number of steps.
    batch_size: Batch size. It is used for validation checks between inputs and
      outputs.
    results: What to return at the end of the aggregation loop.
  """

  def __init__(self, use_steps, num_samples=None, steps=None, batch_size=None):
    self.use_steps = use_steps
    self.num_samples = num_samples
    self.steps = steps
    self.batch_size = batch_size
    self.results = []

  @abc.abstractmethod
  def create(self, batch_outs):
    """Creates the initial results from the first batch outputs.

    Arguments:
      batch_outs: A list of batch-level outputs.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    """Aggregates batch-level results into total results.

    Arguments:
      batch_outs: A list of batch-level outputs.
      batch_start: The start index of this batch. Always `None` if `use_steps`
        is `True`.
      batch_end: The end index of this batch. Always `None` if `use_steps` is
        `True`.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def finalize(self):
    """Prepares the total results to be returned."""
    raise NotImplementedError('Must be implemented in subclasses.')

class SliceAggregator(Aggregator):
  """Combine arrays where the final size is known.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.

  NumPy copies are an operation that threads handle quite well because all of
  the heavy lifting is in c and does not need the GIL. Moreover, we can perform
  lock-free writes to the same buffer in multiple threads because the nature of
  result aggregation guarantees that either the indices are disjoint or the
  aggregator will throw an exception in finalize. Moreover, because aggregation
  is performed on the slowest varying dimension, assignments for a given batch
  will write to contiguous blocks of memory, further minimizing contention.

  There is, however, some scheduling and context switching overhead which will
  offset the gains from pipelining the slice assignment. Below a given threshold
  it is faster to simply assign in the main thread rather than enqueue the
  assigmnet in a side thread. The exact threshold will vary from system to
  system, but the time is not very sensitive to the exact transition so a value
  of 2 ** 14 was chosen which should be reasonable on most systems.
  """

  _BINARY_SIZE_THRESHOLD = 2 ** 14
  _MAX_COPY_SECONDS = 300

  def __init__(self, num_samples, batch_size):
    self._async_copies = []
    self._pool = get_copy_pool()
    self._errors = []
    super(SliceAggregator, self).__init__(
        use_steps=False,
        num_samples=num_samples,
        steps=None,
        batch_size=batch_size)

  def create(self, batch_element):
    # This step does not need to be pipelined because NumPy empty array
    # initialization is effectively instantaneous.
    shape = (self.num_samples,) + batch_element.shape[1:]
    dtype = batch_element.dtype
    if isinstance(batch_element, ops.EagerTensor):
      dtype = dtype.as_numpy_dtype()

    self.results = np.empty(shape=shape, dtype=dtype)

  def aggregate(self, batch_element, batch_start, batch_end):
    # Fail early.
    if self._errors:
      six.reraise(type(self._errors[0]), self._errors[0])

    # In the special case of single batch inference, no copy is needed.
    if batch_end - batch_start == self.num_samples:
      if self.num_samples != batch_element.shape[0]:
        raise ValueError(
            'Mismatch between expected batch size and model output batch size. '
            'Output shape = {}, expected output shape = shape {}'.format(
                batch_element.shape, self.results.shape))

      self.results = batch_element
      return

    # This is an approximate threshold, so we don't need to consider the number
    # of bytes per element.
    num_elements = np.prod(batch_element.shape)
    if num_elements < self._BINARY_SIZE_THRESHOLD:
      self.results[batch_start:batch_end] = batch_element
    else:
      is_finished = threading.Event()
      self._pool.apply_async(
          self._slice_assign,
          args=(batch_element, batch_start, batch_end, is_finished))
      self._async_copies.append(is_finished)

  def _slice_assign(self, batch_element, batch_start, batch_end, is_finished):
    try:
      self.results[batch_start:batch_end] = batch_element

    except Exception as e:  # pylint: disable=broad-except
      # `_slice_assign` should only be called in threads and exceptions raised
      # in threads do not carry over to the main thread. So instead we perform a
      # a broad catch in the thread and then store the exception to be re-raised
      # in the main thread.
      self._errors.append(e)

    finally:
      is_finished.set()

  def finalize(self):
    start_time = time.time()
    for is_finished in self._async_copies:
      timeout = max([0., self._MAX_COPY_SECONDS - (time.time() - start_time)])
      if not is_finished.wait(timeout):
        raise ValueError('Timed out waiting for copy to complete.')

    if self._errors:
      six.reraise(self._errors[0].__class__, self._errors[0])


class OutputsAggregator(Aggregator):
  """Aggregator that concatenates outputs."""

  _structure = None

  def create(self, batch_outs):
    # SparseTensorValue is a named tuple which nest will flatten, so we need
    # to guard it to properly handle the structure.
    self._structure = nest.get_traverse_shallow_structure(
        lambda x: not composite_tensor_utils.is_composite_or_composite_value(x),
        batch_outs)
    batch_outs = nest.flatten_up_to(self._structure, batch_outs)

    for batch_element in batch_outs:
      if composite_tensor_utils.is_composite_or_composite_value(batch_element):
        # If the output is not a ndarray, it will be either a composite tensor
        # or a composite tensor's Value object. In either case, we can't
        # allocate an array to hold the object - we'll handle it later.
        self.results.append(ConcatAggregator(self.batch_size))
      elif isinstance(batch_element, (np.ndarray, ops.EagerTensor)):
        self.results.append(
            (ConcatAggregator(self.batch_size) if self.use_steps else
             SliceAggregator(self.num_samples, self.batch_size)))
      else:
        # This is not a ndarray, a CompositeTensor, or a CompositeTensorValue.
        # Fail fast rather than trying to concatenate it.
        raise RuntimeError('Attempted to aggregate unsupported object {}.'
                           .format(batch_element))

      self.results[-1].create(batch_element)

  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    batch_outs = nest.flatten_up_to(self._structure, batch_outs)
    for batch_element, result in zip(batch_outs, self.results):
      result.aggregate(batch_element, batch_start, batch_end)

  def finalize(self):
    for result in self.results:
      result.finalize()
    self.results = [i.results for i in self.results]
    self.results = nest.pack_sequence_as(self._structure, self.results)


class ConcatAggregator(Aggregator):
  """Combine tensor-likes which cannot be merged on the fly.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.
  """

  def __init__(self, batch_size):
    self.composite = None
    super(ConcatAggregator, self).__init__(
        use_steps=True, num_samples=None, steps=None, batch_size=batch_size)

  def create(self, batch_element):
    self.composite = composite_tensor_utils.is_composite_or_composite_value(
        batch_element)

  def aggregate(self, batch_element, batch_start=None, batch_end=None):

    # TODO(psv): Add num_samples check here to detect when output batch
    # #samples is < batch size and != input batch #samples.
    if self.batch_size and self.batch_size < batch_element.shape[0]:
      raise ValueError(
          'Mismatch between expected batch size and model output batch size. '
          'Output shape = {}, expected output shape = shape {}'.format(
              batch_element.shape,
              (self.batch_size,) + batch_element.shape[1:]))
    self.results.append(batch_element)

  def finalize(self):
    # Special case of single batch inference which skips a copy.
    if len(self.results) == 1:
      self.results = self.results[0]

    elif self.composite:
      # TODO(taylorrobie): efficiently concatenate.
      results = self.results[0]
      for r in self.results[1:]:
        results = composite_tensor_utils.append_composite_tensor(results, r)
      self.results = results

    else:
      self.results = np.concatenate(self.results, axis=0)

    if isinstance(self.results, ops.EagerTensor):
      self.results = self.results._numpy()  # pylint: disable=protected-access


class MetricsAggregator(Aggregator):
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



  gpus_names = [g.name.replace('/physical_device:', '') for g in gpus]
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

def call_metric_function(metric_fn,
                         y_true,
                         y_pred=None,
                         weights=None,
                         mask=None):
  """Invokes metric function and returns the metric result tensor."""
  if mask is not None:
    mask = math_ops.cast(mask, y_pred.dtype)
    if weights is None:
      # Use mask as sample weight.
      weights = mask
    else:
      # Update dimensions of weights to match with mask.
      mask, _, weights = tf_losses_utils.squeeze_or_expand_dimensions(
          mask, sample_weight=weights)
      weights *= mask

  if y_pred is not None:
    return metric_fn(y_true, y_pred, sample_weight=weights)
  # `Mean` metric only takes a single value.
  return metric_fn(y_true, sample_weight=weights)


def collect_per_output_metric_info(metrics,
                                   output_names,
                                   output_shapes,
                                   loss_fns,
                                   is_weighted=False):
  """Maps metric names and functions to model outputs.

  Arguments:
      metrics: a list or a list of lists or a dict of metric functions.
      output_names: a list of the names (strings) of model outputs.
      output_shapes: a list of the shapes (strings) of model outputs.
      loss_fns: a list of the loss functions corresponding to the model outputs.
      is_weighted: Boolean indicating whether the given metrics are weighted.

  Returns:
      A list (one entry per model output) of dicts.
      For instance, if the model has 2 outputs, and for the first output
      we want to compute "binary_accuracy" and "binary_crossentropy",
      and just "binary_accuracy" for the second output,
      the list would look like: `[{
          'acc': binary_accuracy(),
          'ce': binary_crossentropy(),
        }, {
          'acc': binary_accuracy(),
        }]`

  Raises:
      TypeError: if an incorrect type is passed for the `metrics` argument.
  """
  if not metrics:
    return [{} for _ in output_names]

  if isinstance(metrics, list):
    any_sub_list = any(isinstance(m, list) for m in metrics)
    if any_sub_list:
      if len(metrics) != len(output_names):
        raise ValueError('When passing a list of lists as `metrics`, '
                         'it should have one entry per model output. '
                         'The model has ' + str(len(output_names)) +
                         ' outputs, but you passed metrics=' + str(metrics))
      # User has provided a list of len = len(outputs).
      nested_metrics = [generic_utils.to_list(m) for m in metrics]
    else:
      # If it is a single list we then apply all metrics to all outputs.
      if len(output_names) > 1:
        nested_metrics = []
        for _ in output_names:
          nested_metrics.append(
              [metrics_module.clone_metric(m) for m in metrics])
      else:
        nested_metrics = [metrics]
  elif isinstance(metrics, collections.Mapping):
    generic_utils.check_for_unexpected_keys('metrics', metrics, output_names)
    nested_metrics = []
    for name in output_names:
      output_metrics = generic_utils.to_list(metrics.get(name, []))
      nested_metrics.append(output_metrics)
  else:
    raise TypeError('Type of `metrics` argument not understood. '
                    'Expected a list or dictionary, found: ' + str(metrics))

  per_output_metrics = []
  for i, metrics in enumerate(nested_metrics):
    metrics_dict = OrderedDict()
    for metric in metrics:
      metric_name = get_metric_name(metric, is_weighted)
      metric_fn = get_metric_function(
          metric, output_shape=output_shapes[i], loss_fn=loss_fns[i])

      # If the metric function is not stateful, we create a stateful version.
      if not isinstance(metric_fn, metrics_module.Metric):
        metric_fn = metrics_module.MeanMetricWrapper(
            metric_fn, name=metric_name)
      metrics_dict[metric_name] = metric_fn
    per_output_metrics.append(metrics_dict)

  return per_output_metrics


def prepare_loss_functions(loss, output_names):
  """Converts loss to a list of loss functions.

  Arguments:
      loss: String (name of objective function), objective function or
        `tf.losses.Loss` instance. See `tf.losses`. If the model has multiple
        outputs, you can use a different loss on each output by passing a
        dictionary or a list of losses. The loss value that will be minimized by
        the model will then be the sum of all individual losses.
      output_names: List of model output names.

  Returns:
      A list of loss objective functions.

  Raises:
      ValueError: If loss is a dict with keys not in model output names,
          or if loss is a list with len not equal to model outputs.
  """
  if isinstance(loss, collections_abc.Mapping):
    generic_utils.check_for_unexpected_keys('loss', loss, output_names)
    loss_functions = []
    for name in output_names:
      if name not in loss:
        logging.warning(
            'Output {0} missing from loss dictionary. We assume '
            'this was done on purpose. The fit and evaluate APIs will not be '
            'expecting any data to be passed to {0}.'.format(name))
      loss_functions.append(get_loss_function(loss.get(name, None)))
  elif isinstance(loss, six.string_types):
    loss_functions = [get_loss_function(loss) for _ in output_names]
  elif isinstance(loss, collections_abc.Sequence):
    if len(loss) != len(output_names):
      raise ValueError('When passing a list as loss, it should have one entry '
                       'per model outputs. The model has {} outputs, but you '
                       'passed loss={}'.format(len(output_names), loss))
    loss_functions = nest.map_structure(get_loss_function, loss)
  else:
    loss_functions = [get_loss_function(loss) for _ in range(len(output_names))]

  return loss_functions

def generic_output_names(outputs_list):
  return ['output_%d' % (i + 1) for i in range(len(outputs_list))]

def should_run_validation(validation_freq, epoch):
  """Checks if validation should be run this epoch.

  Arguments:
    validation_freq: Integer or list. If an integer, specifies how many training
      epochs to run before a new validation run is performed. If a list,
      specifies the epochs on which to run validation.
    epoch: Integer, the number of the training epoch just completed.

  Returns:
    Bool, True if validation should be run.

  Raises:
    ValueError: if `validation_freq` is an Integer and less than 1, or if
    it is neither an Integer nor a Sequence.
  """
  # `epoch` is 0-indexed internally but 1-indexed in the public API.
  one_indexed_epoch = epoch + 1

  if isinstance(validation_freq, int):
    if validation_freq < 1:
      raise ValueError('`validation_freq` can not be less than 1.')
    return one_indexed_epoch % validation_freq == 0

  if not isinstance(validation_freq, collections_abc.Container):
    raise ValueError('`validation_freq` must be an Integer or '
                     '`collections_abc.Container` (e.g. list, tuple, etc.)')
  return one_indexed_epoch in validation_freq


def get_metric_name(metric, weighted=False):
  """Returns the name corresponding to the given metric input.

  Arguments:
    metric: Metric function name or reference.
    weighted: Boolean indicating if the given metric is weighted.

  Returns:
      The metric name.
  """
  # if tf2.enabled():
    # We keep the string that the user has set in compile as the metric name.
  if isinstance(metric, six.string_types):
    return metric

  metric = metrics_module.get(metric)
  return metric.name if hasattr(metric, 'name') else metric.__name__
  # else:
  # metric_name_prefix = 'weighted_' if weighted else ''
  # if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
  #   if metric in ('accuracy', 'acc'):
  #     suffix = 'acc'
  #   elif metric in ('crossentropy', 'ce'):
  #     suffix = 'ce'
  # else:
  #   metric_fn = metrics_module.get(metric)
  #   # Get metric name as string
  #   if hasattr(metric_fn, 'name'):
  #     suffix = metric_fn.name
  #   else:
  #     suffix = metric_fn.__name__
  # metric_name = metric_name_prefix + suffix
  # return metric_name



def get_metric_function(metric, output_shape=None, loss_fn=None):
  """Returns the metric function corresponding to the given metric input.

  Arguments:
      metric: Metric function name or reference.
      output_shape: The shape of the output that this metric will be calculated
        for.
      loss_fn: The loss function used.

  Returns:
      The metric function.
  """
  if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
    return metrics_module.get(metric)

  is_sparse_categorical_crossentropy = (
      isinstance(loss_fn, losses.SparseCategoricalCrossentropy) or
      (isinstance(loss_fn, losses.LossFunctionWrapper) and
       loss_fn.fn == losses.sparse_categorical_crossentropy))

  is_binary_crossentropy = (
      isinstance(loss_fn, losses.BinaryCrossentropy) or
      (isinstance(loss_fn, losses.LossFunctionWrapper) and
       loss_fn.fn == losses.binary_crossentropy))

  if metric in ['accuracy', 'acc']:
    if output_shape[-1] == 1 or is_binary_crossentropy:
      return metrics_module.binary_accuracy
    elif is_sparse_categorical_crossentropy:
      return metrics_module.sparse_categorical_accuracy
    # If the output_shape[-1] is not 1, then we know output is `categorical`.
    # We assume it is sparse categorical only if loss is explicitly given
    # as sparse categorical crossentropy loss.
    return metrics_module.categorical_accuracy
  else:
    if output_shape[-1] == 1 or is_binary_crossentropy:
      return metrics_module.binary_crossentropy
    elif is_sparse_categorical_crossentropy:
      return metrics_module.sparse_categorical_crossentropy
    return metrics_module.categorical_crossentropy


def get_loss_function(loss):
  """Returns the loss corresponding to the loss input in `compile` API."""
  if loss is None or isinstance(loss, losses.Loss):
    return loss

  # Deserialize loss configuration, if needed.
  if isinstance(loss, collections_abc.Mapping):
    loss = losses.get(loss)

  # Custom callable class.
  if callable(loss) and not hasattr(loss, '__name__'):
    return loss

  # Wrap loss function with signature `(y_true, y_pred, **kwargs)`
  # in `LossFunctionWrapper` class.
  loss_fn = losses.get(loss)

  # For losses which are given as strings/functions in the compile API,
  # we always set the loss reduction type to be `SUM_OVER_BATCH_SIZE`
  # (both in distribution strategy context and otherwise).
  return losses.LossFunctionWrapper(
      loss_fn,
      name=loss_fn.__name__,
      reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)


