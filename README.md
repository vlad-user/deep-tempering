# deep-tempering
Replica-exchange optimisation method for Tensorflow with Keras-like interface.


(Intened usage example)
```python
import random

import tensorflow as tf
import numpy as np

import pt_ensemble

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
  model = tf.keras.models.Model(inputs, res)

  return model

ensemble = pt_ensemble.EnsembleModel(model_builder)
hp = {
  'learning_rate': [0.0 , 0.03],
  'dropout_rate': [0.0, 0.1]
}

# (maybe to put `exchange_hparams` to `fit()`)
ensemble.compile(optimizer=tf.keras.optimizers.SGD(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

x = np.random.normal(0, 1, (10, 2))
y = np.random.randint(0, 2, (10,))
history = ensemble.fit(x,
                       y,
                       exchange_hparams=hp,
                       batch_size=2,
                       epochs=2,
                       swap_step=4,
                       burn_in=15)
```