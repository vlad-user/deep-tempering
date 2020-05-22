import random
import copy

import tensorflow as tf
import numpy as np

from deep_tempering import training
from deep_tempering import training_utils
from deep_tempering import callbacks as cbks
from deep_tempering.training_test import model_builder

def test_configure_callbacks():
  model = training.EnsembleModel(model_builder)
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  n_replicas = 6
  model.compile(optimizer, loss, n_replicas)

  hparams_dict = {
        'learning_rate': np.linspace(0.001, 0.01, n_replicas),
        'dropout_rate': np.linspace(0., 0.6, n_replicas)
  }
  kwargs = {
      'do_validation': True,
      'batch_size': 2,
      'epochs': 2,
      'steps_per_epoch': None,
      'samples': None,
      'verbose': 1,
      'burn_in': None,
      'swap_step': None
  }
  callbacklist = cbks.configure_callbacks([], model, **kwargs)
  kwargs.update({
      'metrics': model.metrics_names + ['val_' + m for m in model.metrics_names]
  })

  kwargs['steps'] = (kwargs['steps_per_epoch'], kwargs.pop('steps_per_epoch'))[0]

  # test that params are stored as intended
  assert kwargs == callbacklist.params

def test_base_hp_exchange_callback():
  tf.compat.v1.keras.backend.clear_session()
  em = training.EnsembleModel(model_builder)
  optimizer = tf.keras.optimizers.SGD()
  loss = 'binary_crossentropy'
  n_replicas = 6
  em.compile(optimizer, loss, n_replicas)
  hparams_dict = {
        'learning_rate': np.linspace(0.001, 0.01, n_replicas),
        'dropout_rate': np.linspace(0., 0.6, n_replicas)
  }
  hpss = training_utils.HyperParamSpace(em, hparams_dict)
  x = np.random.normal(0, 2, (18, 2))
  y = np.random.randint(0, 2, (18, 1))
  clb = cbks.BaseExchangeCallback((x, y), swap_step=10)
  clb.model = em

  # test get_ordered_losses() and _metrics_sorting_key()
  input_ = ['loss_' + str(i) for i in range(n_replicas)]
  input_ = input_ + ['loss_1_' + str(i) for i in range(n_replicas)]
  logs = {l: np.random.uniform() for l in input_}
  expected = copy.deepcopy(input_)
  random.shuffle(input_)
  actual = clb.get_ordered_losses(logs)
  actual = [x[0] for x in actual]
  assert actual == expected

  # test should_exchange property
  em.global_step = 10
  assert clb.should_exchange()

  em.global_step = 9
  assert not clb.should_exchange()

def test_metropolis_callback():
  tf.compat.v1.keras.backend.clear_session()
  em = training.EnsembleModel(model_builder)
  optimizer = tf.keras.optimizers.SGD()
  loss = 'binary_crossentropy'
  n_replicas = 10
  em.compile(optimizer, loss, n_replicas)
  em.global_step = 0
  hparams_dict = {
        'learning_rate': np.linspace(0.001, 0.01, n_replicas),
        'dropout_rate': np.linspace(0.05, 0.6, n_replicas)
  }

  hpspace = training_utils.HyperParamSpace(em, hparams_dict)

  x = np.random.normal(0, 0.2, (18, 2))
  y = np.random.randint(0, 2, (18, 1))
  clb = cbks.MetropolisExchangeCallback((x, y), swap_step=10)
  clb.model = em
  em._hp_state_space = hpspace

  losses = list(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8]))
  hpname = 'dropout_rate'

  # expected state of hyperparams after calling `exchage()` function
  expected = copy.deepcopy(hpspace.hpspace)
  t = expected[8]['dropout_rate']
  expected[8]['dropout_rate'] = expected[9]['dropout_rate']
  expected[9]['dropout_rate'] = t

  # this pair must exchange with probability of one because
  # (beta_i - beta_j) < 0 and losses[i] - losses[j] = 0.8 - 0.9 < 0
  # and exp((beta_i - beta_j) * (losses[i] - losses[j])) > 1
  exchange_pair = 9
  clb.exchange_hyperparams(hpname=hpname, exchange_pair=exchange_pair)
  assert hpspace.hpspace == expected

def test_all_exchange_callback():
  # Add testing here when there are multiple exchange callbacks

  # setting up
  tf.compat.v1.keras.backend.clear_session()
  model = training.EnsembleModel(model_builder)

  n_replicas = 4
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  model.compile(optimizer, loss, n_replicas)
  x = np.random.normal(0, 1, (6, 2))

  y_train = np.arange(6).astype('float')
  y_test = np.arange(6, 12).astype('float')
  hp = {
      'learning_rate': [0.01 , 0.02, 0.03, 0.04],
      'dropout_rate':  [0.1,   0.2,  0.3, 0.4]
  }

  batch_size = 3
  epochs = 5
  do_validation = False
  validation_data = (x, y_test)
  callbacks = []
  samples = 6
  exchange_data = validation_data
  swap_step = 2
  hpss = training_utils.HyperParamSpace(model, hp)
  model._hp_state_space = hpss
  # values of losses that train_on_batch/test_on_batch
  # will return
  def train_on_batch(x, y):
    train_losses = [0.16, 0.15, 0.14, 0.13]
    return train_losses

  def test_on_batch(x, y):
    test_losses = [0.25, 0.24, 0.23, 0.22]
    return test_losses

  model.train_on_batch = train_on_batch
  model.test_on_batch = test_on_batch

  # test that we've added correctly the ExchangeCallback
  callbacks_list = cbks.configure_callbacks(callbacks,
                                            model,
                                            do_validation=do_validation,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            exchange_data=exchange_data,
                                            swap_step=swap_step)
  # callbacks_list has instance of `BaseExchangeCallback`
  assert any(isinstance(c, cbks.BaseExchangeCallback)
             for c in callbacks_list.callbacks)

  def get_first_exchange_callback():
    for cbk in callbacks_list.callbacks:
      if isinstance(cbk, cbks.BaseExchangeCallback):
        return cbk

  prev_hpspace = copy.deepcopy(model.hpspace.hpspace)

  get_first_exchange_callback()._safe_exchange(hpname='dropout_rate',
                                               exchange_pair=3)
  # test that exchange happened
  assert model.hpspace.hpspace[3]['dropout_rate'] == prev_hpspace[2]['dropout_rate']

  get_first_exchange_callback()._safe_exchange(hpname='learning_rate',
                                               exchange_pair=3)
  # test that exchange happened
  assert model.hpspace.hpspace[3] == prev_hpspace[2]
  assert get_first_exchange_callback().exchange_logs['swaped'] == [1, 1]
  assert get_first_exchange_callback().exchange_logs['hpname'] == ['dropout_rate', 'learning_rate']