import random

import numpy as np
import tensorflow as tf

def model_iteration(model,
                    inputs,
                    targets,
                    batch_size=None,
                    epochs=1,
                    callbacks=None):
  
  
  

class HyperParamState:
  """Represents the hyper-parameter state of all replicas."""
  def __init__(self, hp_dict):
    """Creates a new `HyperParamState` instance.
    ```python
    hp_dict = {
        'learning_rate': np.linspace(0.001, 0.01, 6),
        'dropout_rate': np.linspace(0., 0.6, 6)
    }
    hps = HyperParamsState(hp_dict)
    hps._hpgrid
    # {0: {'learning_rate': 0.001, 'dropout_rate': 0.0},
    #  1: {'learning_rate': 0.0055000000000000005, 'dropout_rate': 0.3},
    #  2: {'learning_rate': 0.01, 'dropout_rate': 0.6}}
    ```
    """

    hp_dict = dict((k, list(v)) for k, v in hp_dict.items())
    n_replicas = len(hp_dict[hp_dict.__iter__().__next__()])
    self.n_replicas = n_replicas
    self._hpgrid = {
        i: {k: v[i] for k, v in hp_dict.items()}
        for i in range(n_replicas)
    }

  def swap_between(self, replica_i, replica_j, hp_name):
    hp_i = self._hpgrid[replica_i][hp_name]
    hp_j = self._hpgrid[replica_j][hp_name]
    self._hpgrid[replica_j][hp_name] = hp_i
    self._hpgrid[replica_i][hp_name] = hp_j

  def get_ordered_hparams(self, hp_name):
    hparams = [(i, self._hpgrid[hp_name][i]) for i in range(self.n_replicas)]
    hparams.sort(key=lambda x: x[1])
    return [hp[1] for hp in hparams]

  def __getattr__(self, key):
    if key == 'hpgrid':
      return self._hpgrid.copy()

class HyperParams:
  self._attrs = {}

  def __getattr__(self, key):
    if key in self._attrs:
      return self._attrs[key]
    if key == 'dropout_rate':
      dropout_rate = tf.compat.v1.placeholder_with_default(0, (), 'dropout')
      self._attrs[key] = dropout_rate
      return dropout_rate
    if key in {'learning_rate', 'lr'}:
      lr = tf.compat.v1.placeholder(tf.float32, shape=(), name='lr')
      self._attrs['learning_rate'] = lr
      self._attrs['lr'] = lr
      return lr

    raise AttributeError('HyperParams does not have attribute', key)



class PTOptimization:

  def __init__(self,
               model_or_model_builder,
               optimizer,
               noise_list,
               noise_types,
               compute_proba_fn=None):
    self.n_replicas = len(noise_list)
    self.models = [keras_model]
    self.models += [tf.keras.models.clone_model(keras_model)
                    for _ in range(n_replicas)]
    self.hp_state = HyperParamsState(noise_list,noise_types)

  def maybe_swap_replicas(self, logs):
    losses = [logs[i]['loss'] for i in range(self.n_replicas)]
    noise_type = self.get_hp_name_to_swap()
    random_pair = self.get_
    i = 
    j = 
    # ...

    return self.hp_state.swap_between(i, j, noise_type)

  def train_on_batch(self, x, y):
    logs = self._parallel_train_on_batch(x, y)
    self.maybe_swap_replicas(logs)
    self.log_batch(logs, hp_state)

  def get_hp_name_to_swap(self):
    """Override this function to get a different scheduling of hparams to swap."""
    hp_names = list(self.hp_state.hpgrid.keys())
    return random.choice(hp_names)

  def get_pair_to_swap(self):
    """Override this function to get a different scheduling of pairs to swap."""
    return random.choice(list(range(self.n_replicas - 1)))