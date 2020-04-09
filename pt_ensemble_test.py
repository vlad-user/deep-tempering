import pytest
import tensorflow as tf
import numpy as np

import pt_ensemble

def test_hp_space_state():

  em = pt_ensemble.EnsembleModel(model_builder)
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  n_replicas = 6
  em.compile(optimizer, loss, n_replicas)

  hparams_dict = {
          'learning_rate': np.linspace(0.001, 0.01, n_replicas),
          'dropout_rate': np.linspace(0., 0.6, n_replicas)
      }
  hpss = pt_ensemble.HPSpaceState(em, hparams_dict)

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

  assert actual == expected

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
  model = tf.keras.models.Model(inputs, res)

  return model

def test_pt_ensemble():

  ensemble = pt_ensemble.EnsembleModel(model_builder)

  # all args are `None`
  optimizers = [None]
  losses = [None]
  exchange_hparams = [None]
  errors = [ValueError]

  # # hyperparams of different sizes
  # optimizers.append(None)
  # losses.append(None)
  # hp = {'learning_rate': [0.0 , 0.03], 'dropout_rate': [0.0]}
  # exchange_hparams.append(hp)
  # errors.append(AssertionError)

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
  return ensemble.fit(x, y, exchange_hparams=hp)
