import pytest
import tensorflow as tf
import numpy as np

from deep_tempering import training

def test_hp_space_state():
  tf.compat.v1.reset_default_graph()
  tf.compat.v1.keras.backend.clear_session()
  em = training.EnsembleModel(model_builder)
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  n_replicas = 6
  em.compile(optimizer, loss, n_replicas)

  hparams_dict = {
          'learning_rate': np.linspace(0.001, 0.01, n_replicas),
          'dropout_rate': np.linspace(0., 0.6, n_replicas)
      }
  hpss = training.HPSpaceState(em, hparams_dict)

  # test that initial hyper-parameter values are correct
  initial_values = {
      0: {'learning_rate': 0.001, 'dropout_rate': 0.0},
      1: {'learning_rate': 0.0028000000000000004, 'dropout_rate': 0.12},
      2: {'learning_rate': 0.0046, 'dropout_rate': 0.24},
      3: {'learning_rate': 0.0064, 'dropout_rate': 0.36},
      4: {'learning_rate': 0.0082, 'dropout_rate': 0.48},
      5: {'learning_rate': 0.01, 'dropout_rate': 0.6}
  }
  assert initial_values == hpss.hpspace

  # swap replica learning rate, replicas 0, 1
  replica_i = 0
  replica_j = 1
  hpss.swap_between(replica_i, replica_j, 'learning_rate')
  expected_values = {
      0: {'learning_rate': 0.0028000000000000004, 'dropout_rate': 0.0},
      1: {'learning_rate': 0.001, 'dropout_rate': 0.12},
      2: {'learning_rate': 0.0046, 'dropout_rate': 0.24},
      3: {'learning_rate': 0.0064, 'dropout_rate': 0.36},
      4: {'learning_rate': 0.0082, 'dropout_rate': 0.48},
      5: {'learning_rate': 0.01, 'dropout_rate': 0.6}
  }
  assert hpss.hpspace == expected_values

  # test the the ordered values are represent adjacent temperatures.
  expected_values = [(1, 0.001),
                     (0, 0.0028000000000000004),
                     (2, 0.0046),
                     (3, 0.0064),
                     (4, 0.0082),
                     (5, 0.01)]
  assert hpss.get_ordered_hparams('learning_rate') == expected_values

  # test that placeholders are correctly fed
  feed_dict = hpss.prepare_feed_tensors_and_values() 

  lr_feed_dict = {k: v for k, v in feed_dict.items() if 'learning_rate' in k.name}
  lr_items = list(lr_feed_dict.items())
  lr_items.sort(key=lambda x: x[0].name)
  actual = [v[1] for v in lr_items]
  expected_values.sort(key=lambda x: x[0])
  expected = [v[1] for v in expected_values]

  print(actual)
  print(lr_items)
  print(expected)
  print(expected_values)
  np.testing.assert_almost_equal(actual, expected)

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
  model = tf.keras.models.Model(inputs, res)

  return model

def test_model_iteration():
  # test that history stores accurate losses
  model = training.EnsembleModel(model_builder)

  n_replicas = 3
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  model.compile(optimizer, loss, n_replicas)
  x = np.random.normal(0, 1, (6, 2))

  y_train = np.arange(6).astype('float')
  y_test = np.arange(6, 12).astype('float')
  hp = {
    'learning_rate': [0.01 , 0.02, 0.3],
    'dropout_rate': [0.0, 0.1, 0.3]
  }

  batch_size = 3
  epochs = 5
  validation_data = (x, y_test)

  def train_on_batch(x, y):
      return [y[0], y[1], y[2]]

  def test_on_batch(x, y):
      return [y[0], y[1], y[2]]

  model.train_on_batch = train_on_batch
  model.test_on_batch = test_on_batch

  history = model.fit(x,
                      y_train,
                      exchange_hparams=hp,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=validation_data,
                      shuffle=False,
                      verbose=0)

  expected_hist = {
      'loss_0': [1.5] * epochs,
      'loss_1': [2.5] * epochs,
      'loss_2': [3.5] * epochs,
      'val_loss_0': [7.5] * epochs,
      'val_loss_1': [8.5] * epochs,
      'val_loss_2': [9.5] * epochs
  }
  assert expected_hist == history.history


  # test the case when the last batch size is smaller than others
  model = training.EnsembleModel(model_builder)
  n_replicas = 3
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  model.compile(optimizer, loss, n_replicas)
  x = np.random.normal(0, 1, (5, 2))
  y_train = np.arange(5).astype('float')
  y_test = np.arange(5, 10).astype('float')
  hp = {
      'learning_rate': [0.01 , 0.02, 0.3],
      'dropout_rate': [0.0, 0.1, 0.3]
  }

  batch_size = 3
  epochs = 5
  validation_data = (x, y_test)


  def train_on_batch(x, y):
    if y.shape[0] < 3:
      res = [y[0], y[1], (y[0] + y[1]) / 2]
    else:
      res = [y[0], y[1], y[2]]
    return res

  def test_on_batch(x, y):
    if y.shape[0] < 3:
      res = [y[0], y[1], (y[0] + y[1]) / 2]
    else:
      res = [y[0], y[1], y[2]]
    return res
  
  model.train_on_batch = train_on_batch
  model.test_on_batch = test_on_batch

  history = model.fit(x,
                      y_train,
                      exchange_hparams=hp,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=validation_data,
                      shuffle=False)

  expected_hist = {
      'loss_0': [0 * (3/5) + 3 * (2/5)] * epochs,
      'loss_1': [1 * (3/5) + 4 * (2/5)] * epochs,
      'loss_2': [2 * (3/5) + 3.5 * (2/5)] * epochs,
      'val_loss_0': [5 * (3/5) + 8 * (2/5)] * epochs,
      'val_loss_1': [6 * (3/5) + 9 * (2/5)] * epochs,
      'val_loss_2': [7 * (3/5) + 8.5 * (2/5)] * epochs
  }
  for k in expected_hist:
      np.testing.assert_almost_equal(history.history[k], expected_hist[k])


def test_pt_ensemble():

  ensemble = training.EnsembleModel(model_builder)

  # all args are `None`
  optimizers = [None]
  losses = [None]
  exchange_hparams = [None]
  errors = [ValueError]

  zipped = zip(optimizers, losses, exchange_hparams, errors)
  for optimizer, loss, hp, error in zipped:
    with pytest.raises(error):
      ensemble.compile(optimizer, loss, n_replicas=2)

  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  ensemble.compile(optimizer, loss, 2)
  x = np.random.normal(0, 1, (10, 2))
  y = np.random.randint(0, 2, (10,))
  hp = {
      'learning_rate': [0.0 , 0.03],
      'dropout_rate': [0.0, 0.1]
  }
  return ensemble.fit(x, y, exchange_hparams=hp, epochs=3, batch_size=2)
