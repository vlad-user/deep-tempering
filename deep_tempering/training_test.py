import pytest
import copy

import tensorflow as tf
import numpy as np
from sklearn.datasets import make_blobs

from deep_tempering import training
from deep_tempering import training_utils
from deep_tempering import callbacks as cbks

def test_hp_space_state():
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
  hpss = training_utils.HyperParamSpace(em, hparams_dict)

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

  np.testing.assert_almost_equal(actual, expected)

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(res)
  model = tf.keras.models.Model(inputs, res)

  return model


def test_model_iteration_without_exchanges():
  # test that history stores accurate losses
  tf.compat.v1.keras.backend.clear_session()
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
      'dropout_rate':  [0.0,   0.1,  0.3]
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
                      hyper_params=hp,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=validation_data,
                      shuffle=False,
                      verbose=0)

  expected_hist = {
      'loss_0':     [1.5] * epochs,
      'loss_1':     [2.5] * epochs,
      'loss_2':     [3.5] * epochs,
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
      'dropout_rate': [0.0001, 0.1, 0.3]
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
                      hyper_params=hp,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=validation_data,
                      shuffle=False)

  expected_hist = {
      'loss_0':     [0 * (3/5) + 3   * (2/5)] * epochs,
      'loss_1':     [1 * (3/5) + 4   * (2/5)] * epochs,
      'loss_2':     [2 * (3/5) + 3.5 * (2/5)] * epochs,
      'val_loss_0': [5 * (3/5) + 8   * (2/5)] * epochs,
      'val_loss_1': [6 * (3/5) + 9   * (2/5)] * epochs,
      'val_loss_2': [7 * (3/5) + 8.5 * (2/5)] * epochs
  }
  for k in expected_hist:
    np.testing.assert_almost_equal(np.squeeze(history.history[k]),
                                   expected_hist[k])

def test_metrics_and_losses():
  """Tests metrics and losses for `model.fit()` and `model.evaluate()`."""
  # Explanation of how do I test metrics and losses:
  # 1. I generate two replicas and train them within the my model_iteration.
  # 2. I train two same models initialized with exact same weights but using
  #   Keras' API.
  # 3. I compare final weights and history values of metrics and losses.

  # test a number of times. 
  for _ in range(1):

    tf.compat.v1.keras.backend.clear_session()
    x_data, y_data = make_blobs(n_samples=32, centers=[[1, 1], [-1, -1]])
    batch_size = 8
    epochs = 5
    verbose = 1
    init1 = np.random.normal(0, 0.5, (2, 2)).astype('float32')
    init2 = np.random.normal(0, 0.2, (2, 1)).astype('float32')

    def init_fn1(*args, **kwargs):
      return init1
    def init_fn2(*args, **kwargs):
      return init2
    def model_builder2(*args):
      inputs = tf.keras.layers.Input((2,))
      res = tf.keras.layers.Dense(2,
                                  activation=tf.nn.relu,
                                  kernel_initializer=init_fn1)(inputs)
      res = tf.keras.layers.Dense(1,
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=init_fn2)(res)
      model = tf.keras.models.Model(inputs, res)
      return model

    metrics = ['accuracy',
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(),
               tf.keras.metrics.AUC(curve='PR'),
               tf.keras.metrics.AUC(curve='ROC'),
               ]
    model = model_builder2()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.003),
                  loss='binary_crossentropy',
                  metrics=metrics)

    hist = model.fit(x_data,
                     y_data,
                     batch_size=batch_size,
                     epochs=epochs,
                     shuffle=False,
                     verbose=verbose)

    expected_loss = hist.history['loss']
    expected_acc = hist.history['acc']
    expected_precision = hist.history['precision']
    expected_recall = hist.history['recall']
    expected_auc = hist.history['auc']
    expected_auc_1 = hist.history['auc_1']

    expected_evaluated = model.evaluate(x_data, y_data, verbose=0)
    expected_predicted = model.predict(x_data)

    # ensemble model
    tf.compat.v1.keras.backend.clear_session()

    metrics = ['accuracy',
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(),
               tf.keras.metrics.AUC(curve='PR'),
               tf.keras.metrics.AUC(curve='ROC'),
               ]

    em = training.EnsembleModel(model_builder2)
    em.compile(optimizer=tf.keras.optimizers.SGD(0.0),
               loss='binary_crossentropy',
               n_replicas=2,
               metrics=metrics)
    hp = {
        'learning_rate': [0.003 , 0.],
    }
    hist2 = em.fit(x_data,
                   y_data,
                   hyper_params=hp,
                   epochs=epochs,
                   batch_size=batch_size,
                   shuffle=False,
                   verbose=verbose)
    actual_evaluated = em.evaluate(x_data, y_data, verbose=0)
    actual_predicted = em.predict(x_data)

    # compare evaluation metrics
    size = len(expected_evaluated)
    np.testing.assert_almost_equal(expected_evaluated,
                                   actual_evaluated[::2][:size])

    # compare predicted outputs
    np.testing.assert_almost_equal(expected_predicted,
                                   actual_predicted[::2][0])

    # compare training history
    loss_0 = hist2.history['loss_0']
    loss_1 = hist2.history['loss_1']
    acc_0 = hist2.history['acc_0']
    acc_1 = hist2.history['acc_1']
    precision_0 = hist2.history['precision_0']
    precision_1 = hist2.history['precision_1']
    recall_0 = hist2.history['recall_0']
    recall_1 = hist2.history['recall_1']
    auc_0 = hist2.history['auc_0']
    auc_1 = hist2.history['auc_1']
    auc_1_0 = hist2.history['auc_1_0']
    auc_1_1 = hist2.history['auc_1_1']

    np.testing.assert_almost_equal(loss_0, expected_loss)
    np.testing.assert_almost_equal(acc_0, expected_acc)
    np.testing.assert_almost_equal(precision_0, expected_precision)
    np.testing.assert_almost_equal(recall_0, expected_recall)
    np.testing.assert_almost_equal(auc_0, expected_auc)
    np.testing.assert_almost_equal(auc_1_0, expected_auc_1)


    # learning rate is 0 - no change is expected
    assert len(set(loss_1)) == 1
    assert len(set(acc_1)) == 1
    assert len(set(precision_1)) == 1
    assert len(set(recall_1)) == 1
    assert len(set(auc_1)) == 1
    assert len(set(auc_1_1)) == 1

    # test that the extraction of replica (keras model) that corresponds to the
    # minimal loss is correct
    optimal_model = em.optimal_model()
    sess = tf.compat.v1.keras.backend.get_session()
    graph = sess.graph
    optimal_model.compile(optimizer=tf.keras.optimizers.SGD(),
                          loss='binary_crossentropy')

    optimal_loss = optimal_model.evaluate(x_data, y_data)
    min_loss = optimal_loss
    evaluated_losses = em.evaluate(x_data, y_data)[:em.n_replicas]
    np.testing.assert_almost_equal(min_loss, min(evaluated_losses))

  tf.compat.v1.keras.backend.clear_session()


def ensemble_model_predict(model, data):
  predicted = []
  for i in range(model.n_replicas):
    predicted.append(model._train_attrs[i]['model'].predict(data))
  return predicted

def test_pt_ensemble():

  ensemble = training.EnsembleModel(model_builder)

  # all args are `None`
  optimizers = [None]
  losses = [None]
  hyper_params = [None]
  errors = [ValueError]

  zipped = zip(optimizers, losses, hyper_params, errors)
  for optimizer, loss, hp, error in zipped:
    with pytest.raises(error):
      ensemble.compile(optimizer, loss, n_replicas=2)

  optimizer = tf.keras.optimizers.SGD()
  loss = 'categorical_crossentropy'
  metrics = ['accuracy', tf.keras.metrics.Precision()]
  ensemble.compile(optimizer, loss, 2, metrics=metrics)
  x = np.random.normal(0, 1, (10, 2))
  y = np.random.randint(0, 2, (10, 1))
  hp = {
      'learning_rate': [0.0 , 0.03],
      'dropout_rate': [0.0, 0.1]
  }
  return ensemble.fit(x, y, hyper_params=hp, epochs=3, batch_size=2)
