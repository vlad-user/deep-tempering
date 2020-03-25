import pytest
import tensorflow as tf
import numpy as np

import pt_ensemble

def test_pt_ensemble():

  def model_builder(hp):
    inputs = tf.keras.layers.Input((2,))
    res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
    res = tf.keras.layers.Dropout(dropout_rate)(res)
    res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
    model = tf.keras.models.Model(inputs, res)

    return model

  ensemble = pt_ensemble.PTEnsemble(model_builder)

  # all args are `None`
  optimizers = [None]
  losses = [None]
  exchange_hparams = [None]
  errors = [ValueError]

  # hyperparams of different sizes
  optimizers.append(None)
  losses.append(None)
  hp = {'learning_rate': [0.0 , 0.03], 'dropout_rate': [0.0]}
  exchange_hparams.append(hp)
  errors.append(AssertionError)
  
  zipped = zip(optimizers, losses, exchange_hparams, errors)
  for optimizer, loss, hp, error in zipped:
    with pytest.raises(error):
      ensemble.compile(optimizer, loss, hp)

  hp = {'learning_rate': [0.0 , 0.03], 'dropout_rate': [0.0, 0.1]}
  #optimizer = tf.keras.optimizers.SGD()
  optimizer = 'sgd'
  optimizer = tf.keras.optimizers.SGD()
  loss = 'sparse_categorical_crossentropy'
  ensemble.compile(optimizer, loss, hp)
  x = np.random.normal(0, 1, (10, 2))
  y = np.random.randint(0, 2, (10,))
  return ensemble.fit(x, y)

