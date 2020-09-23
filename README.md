# deep-tempering
Replica-exchange optimisation method for Tensorflow with Keras-like interface.

## Installation
```
pip install git+https://github.com/vlad-user/deep-tempering.git
```

## Example 1:
```python
import tensorflow as tf
import numpy as np

import deep_tempering as dt

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
  model = tf.keras.models.Model(inputs, res)

  return model

n_replicas = 6
model = dt.EnsembleModel(model_builder)
hp = {
    'learning_rate': np.linspace(0.01, 0.001, n_replicas),
    'dropout_rate': np.linspace(0, 0.5, n_replicas)
}

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=n_replicas)

x = np.random.normal(0, 1, (10, 2))
y = np.random.randint(0, 2, (10,))

history = model.fit(x,
                    y,
                    hyper_params=hp,
                    batch_size=2,
                    epochs=2,
                    swap_step=4,
                    burn_in=15)

# access the optimal (not compiled) keras' model instance
optimal_model = model.optimal_model()

# inference only on the trained optimal model
predicted = optimal_model.predict(x)
```
## Example 2: using scheduled hyper-parameters
```python
import tensorflow as tf
import numpy as np

import deep_tempering as dt

tf.compat.v1.keras.backend.clear_session()

def model_builder(hp):
  inputs = tf.keras.layers.Input((2,))
  res = tf.keras.layers.Dense(2, activation=tf.nn.relu)(inputs)
  dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)
  res = tf.keras.layers.Dropout(dropout_rate)(res)
  res = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(res)
  model = tf.keras.models.Model(inputs, res)

  return model

n_replicas = 6
model = dt.EnsembleModel(model_builder)

hp = {
    1: {'learning_rate': np.ones((n_replicas, )) * 0.1, # starting at step 1
        'dropout_rate': np.zeros((n_replicas,))},
    10: {'learning_rate': np.linspace(0.01, 0.001, n_replicas), # starting at step 200
          'dropout_rate': np.linspace(0, 0.5, n_replicas)}
}

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=n_replicas)

x = np.random.normal(0, 1, (20, 2))
y = np.random.randint(0, 2, (20,))

history = model.fit(x,
                    y,
                    hyper_params=hp,
                    batch_size=2,
                    epochs=15,
                    swap_step=3,
                    callbacks=[dt.callbacks.MetropolisExchangeCallback((x, y), 3, 5)],
                    burn_in=0,
                    verbose=1)

metrics_history = history.history
exchange_history = history.exchange_history
```