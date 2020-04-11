import numpy as np

from . import train_utils as utils


def test_iterator():
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

def test_prepare_data_iterables():
  data_size = 10
  x_data = np.random.normal(0, 1, size=(data_size, 2))
  y_data = np.random.randint(0, 2, size=(data_size,))

  # test that 1 dataset is returned when there is no validation split
  datasets = utils.prepare_data_iterables(x_data, y_data)
  assert len(datasets) == 1
  
  # test that 2 datasets are returns when 1 > validation_split > 0
  validation_split = 0.2
  datasets = utils.prepare_data_iterables(x_data,
                                          y_data,
                                          validation_split=validation_split)
  assert len(datasets) == 2

  # test that the test_size == validation_split * data_size and that
  #               train_size == data_size - test_size and that
  #               x_test.shape[0] == y_test.shape[0]
  # (in that order)
  validation_split = 0.2
  datasets = utils.prepare_data_iterables(x_data,
                                          y_data,
                                          validation_split=validation_split,
                                          shuffle=False)
  train_dataset, test_dataset = datasets
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