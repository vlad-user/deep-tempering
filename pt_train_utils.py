import tensorflow as tf
import numpy as np
from tensorflow.python.keras import callbacks as cbks
from sklearn.model_selection import train_test_split

def infer_shape_from_numpy_array(ary):
  if len(ary.shape) == 1:
    return (None,)
  return (None,) + ary.shape[1:]

def create_training_target(shape, dtype=None):
  dtype = dtype or tf.int32

  if shape[0] is None:
    shape = shape[1:]

  return tf.keras.layers.Input(shape, dtype=dtype)

class Logger:
  pass

def prepare_data_iterables(x,
                           y,
                           validation_split=0.0,
                           validation_data=None,
                           batch_size=32,
                           epochs=1,
                           shuffle=True,
                           shuffle_buf_size=1024,
                           random_state=0):
  if validation_split == 0.0 and validation_data is None:
    return [DataIterable(x, y, batch_size, epochs, shuffle, shuffle_buf_size)]

  elif validation_split == 0.0 and validation_data is not None:
    train_dataset = DataIterable(x,
                                 y,
                                 batch_size,
                                 epochs,
                                 shuffle,
                                 shuffle_buf_size)
    test_dataset = DataIterable(validation_data[0],
                                validation_data[1],
                                batch_size,
                                epochs,
                                shuffle,
                                shuffle_buf_size)
    return [train_dataset, test_dataset]

  elif  0.0 < validation_split < 1:
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=validation_split, random_state=random_state)
    train_dataset = DataIterable(x_train,
                                 y_train,
                                 batch_size,
                                 epochs,
                                 shuffle,
                                 shuffle_buf_size)
    test_dataset = DataIterable(x_test,
                                y_test,
                                batch_size,
                                epochs,
                                shuffle,
                                shuffle_buf_size)
    return [train_dataset, test_dataset]
  else:
    raise ValueError('Cannot parition data.')

class DataIterable:
  # TODO: Extend support for eager iteration
  # Implement Wrapper for other than numpy arrays data types.

  def __init__(self, x, y, batch_size=32, epochs=1, shuffle=True, shuffle_buf_size=1024):
    self.x = x
    self.y = y
    self.batch_size = min(batch_size, y.shape[0])
    self.epochs = epochs
    self.shuffle = shuffle
    self.shuffle_buf_size = shuffle_buf_size
    self.__len = x.shape[0]
    d = tf.data.Dataset.from_tensor_slices(
        {
            'x': self.x,
            'y': self.y
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
