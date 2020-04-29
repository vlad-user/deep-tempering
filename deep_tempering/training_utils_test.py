import copy

import numpy as np
from sklearn.model_selection import train_test_split
import pytest

from deep_tempering import training_utils as utils


def test_graph_mode_iterator():
  data_size = 10
  x_data = np.random.normal(0, 1, size=(data_size, 1))
  y_data = np.arange(data_size)
  batch_size = 2
  d = utils.GraphModeDataIterable(x_data,
                         y_data,
                         batch_size=batch_size,
                         epochs=2,
                         shuffle=False)
  i = 0
  for (x, y) in d:
    if i >= data_size:
        i = 0
    assert np.all(y == y_data[i: i + batch_size])
    i += batch_size

def test_numpy_iterator():
  # test without shuffling
  data_size = 10
  x_data = np.random.normal(0, 1, size=(data_size, 1))
  y_data = np.arange(data_size)
  batch_size = 2
  d = utils.DataIterable(x_data,
                              y_data,
                              batch_size=batch_size,
                              epochs=2,
                              shuffle=False)
  i = 0
  for (x, y) in d:
    if i >= data_size:
        i = 0
    assert np.all(y == y_data[i: i + batch_size])
    i += batch_size

  # test with shuffling, data_size mod batch_size = 0
  data_size = 16
  batch_size = 8
  x_data = np.random.normal(0, 1, size=(16, 1))
  y_data = np.arange(16)

  x_data2 = copy.deepcopy(x_data)
  y_data2 = copy.deepcopy(y_data)

  iterable = utils.DataIterable(x_data2,
                                y_data2,
                                batch_size=8,
                                shuffle=True,
                                epochs=1)
  result_x, result_y = [], []

  for (x, y) in iterable:
      result_x.append(x)
      result_y.append(y)

  result_x = list(np.concatenate(result_x, 0))
  result_y = list(np.concatenate(result_y, 0))

  sorted_y, sorted_x = zip(*sorted(zip(result_y, result_x)))

  np.testing.assert_almost_equal(x_data, sorted_x)
  np.testing.assert_almost_equal(y_data, sorted_y)

  # test with shuffling, data_size mod batch_size != 0
  data_size = 37
  batch_size = 8
  x_data = np.random.normal(0, 1, size=(16, 1))
  y_data = np.arange(16)

  x_data2 = copy.deepcopy(x_data)
  y_data2 = copy.deepcopy(y_data)

  iterable = utils.DataIterable(x_data2,
                                y_data2,
                                batch_size=8,
                                shuffle=True,
                                epochs=1)
  result_x, result_y = [], []

  for (x, y) in iterable:
      result_x.append(x)
      result_y.append(y)

  result_x = list(np.concatenate(result_x, 0))
  result_y = list(np.concatenate(result_y, 0))

  sorted_y, sorted_x = zip(*sorted(zip(result_y, result_x)))
  sorted_y = np.array(sorted_y)
  sorted_x = np.vstack(sorted_x)

  np.testing.assert_almost_equal(x_data, sorted_x)
  np.testing.assert_almost_equal(y_data, sorted_y)



def test_prepare_data_iterables():
  data_size = 10
  x_data = np.random.normal(0, 1, size=(data_size, 2))
  y_data = np.random.randint(0, 2, size=(data_size,))

  # test that 1 dataset is returned when there is no validation split
  datasets = utils.prepare_data_iterables(x_data, y_data)
  print(datasets)

  # test that 2 datasets are returned when 1 > validation_split > 0
  validation_split = 0.2
  datasets = utils.prepare_data_iterables(x_data,
                                          y_data,
                                          validation_split=validation_split)
  assert len(datasets) == 3

  # test that the test_size == validation_split * data_size and that
  #               train_size == data_size - test_size and that
  #               x_test.shape[0] == y_test.shape[0]
  # (in that order)
  validation_split = 0.2
  datasets = utils.prepare_data_iterables(x_data,
                                          y_data,
                                          validation_split=validation_split,
                                          shuffle=False)
  train_dataset, test_dataset, _ = datasets
  print(train_dataset, test_dataset, _)
  results_x = []
  results_y = []
  for (x, y) in test_dataset:
      results_x.append(x)
      results_y.append(y)
  x_test = np.concatenate(results_x, axis=0)
  y_test = np.concatenate(results_y, axis=0)

  results_x = []
  results_y = []
  for (x, y) in train_dataset:
      results_x.append(x)
      results_y.append(y)
  x_train = np.concatenate(results_x, axis=0)
  y_train = np.concatenate(results_y, axis=0)

  assert x_test.shape[0] == validation_split * data_size
  assert x_train.shape[0] == data_size - validation_split * data_size
  assert x_test.shape[0] == y_test.shape[0]

def test_train_validation_exchange_data():
  """Test the splits between train, validation and exchange data."""
  x_train_input = np.random.normal(0, 2, (100, 2))
  y_train_input = np.random.randint(0, 2, (100,))
  train_data_input = (x_train_input, y_train_input)
  
  validation_data_input = (np.random.normal(0, 1, (20, 2)),
                           np.random.randint(0, 2, (20)))
  exchange_data_input = (np.random.normal(0, 1, (20, 2)),
                         np.random.randint(0, 2, (20)))

  # test that args must be ndarrays
  with pytest.raises(AssertionError):
    utils._train_validation_exchange_data(
        (x_train_input, list(y_train_input)), None, None)

  def assert_almost_equal(iter1, iter2):
      for ary1, ary2 in zip(iter1, iter2):
          np.testing.assert_almost_equal(ary1, ary2)

  # validation_data and exchange_data are passed explicitly
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, validation_data_input, exchange_data_input))
  assert_almost_equal(train_data + validation_data + exchange_data,
                      train_data_input + validation_data_input + exchange_data_input)

  # validation_data is None, exchange_data is not None
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, None, exchange_data_input))
  assert_almost_equal(train_data + exchange_data, train_data_input + exchange_data_input)
  assert validation_data is None
  
  # validation data is not None, exchange data is None
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, validation_data_input, None))
  assert_almost_equal(train_data + exchange_data + validation_data,
                      train_data_input + validation_data_input + validation_data_input)
  
  # validation_data is None, but validation_split == 0.2
  random_state = 0
  validation_split = 0.2
  x, x_valid, y, y_valid = train_test_split(*train_data_input,
                                            random_state=random_state,
                                            test_size=validation_split)
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, validation_split=validation_split))
  assert_almost_equal(train_data + exchange_data + validation_data,
                      (x, y, x_valid, y_valid, x_valid, y_valid))
  
  # validation_data is None, validation_split == 0., exchange_split == 0.3
  random_state = 0
  exchange_split = 0.3
  x, x_exchange, y, y_exchange = train_test_split(*train_data_input,
                                            random_state=random_state,
                                            test_size=exchange_split)
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, exchange_split=exchange_split))
  assert_almost_equal(train_data + exchange_data,
                      (x, y, x_exchange, y_exchange))
  assert validation_data is None
  
  # validation_data is None, exchange_data is None,
  # validation_split == 0.1, exchange_split = 0.15
  random_state = 0
  exchange_split = 0.15
  validation_split = 0.1
  x, x_valid, y, y_valid = train_test_split(*train_data_input,
                                            random_state=random_state,
                                            test_size=validation_split)
  x, x_exchange, y, y_exchange = train_test_split(*(x, y),
                                                  random_state=random_state,
                                                  test_size=exchange_split)
  train_data, validation_data, exchange_data = (
      utils._train_validation_exchange_data(
          train_data_input, validation_split=validation_split,
          exchange_split=exchange_split))
  assert_almost_equal(train_data + validation_data + exchange_data,
                      (x, y) + (x_valid, y_valid) + (x_exchange, y_exchange))
