import tensorflow as tf
from tensorflow.python.keras.engine import training_utils as keras_train_utils
from tensorflow.python.keras.engine import training as keras_train
from tensorflow.python.keras import optimizers as keras_optimizers
import numpy as np

class HyperParams:
  def __init__(self):
    self._attrs = {}

  def get_hparam(self, hp_name, default_value=None):

    if hp_name in self._attrs:
      raise ValueError('Hyper Params with name ', hp_name, 'already exists.')

    if default_value is None:
      hp = tf.compat.v1.placeholder(tf.float32, shape=(), name=hp_name)
    else:
      hp = tf.compat.v1.placeholder_with_default(default_value, shape=(), name=hp_name)

    self._attrs[hp_name] = hp
    return hp

class PTEnsemble:
  """Mimics the behaviour of `keras.Model` for ensemble PT training."""
  def __init__(self, model_builder):
    """Instantiates a new PTEnsemble instance."""
    if not callable(model_builder):
      raise TypeError("Expected callable `model_builder`.")
    self._model_builder_fn = model_builder
    self._is_compiled = False
    self.run_eagerly = False

  def compile(self,
              optimizer,
              loss,
              exchange_hparams,
              target_tensors=None):
    
    # verify valid `exchange_hparams`
    if not isinstance(exchange_hparams, dict):
      raise ValueError("`exchange_hparams must be an instance of `dict`.")
    exchange_hparams = {h: np.array(v) for h, v in exchange_hparams.items()}
    unique_lens = set(v.shape[0] for v in exchange_hparams.values())
    assert len(unique_lens) == 1

    # validate losses
    # ...

    # validate optimizer
    # ...

    self.n_replicas = list(unique_lens)[0]

    train_attrs = {i: {} for i in range(self.n_replicas)}
    for i in range(self.n_replicas):

      with tf.variable_scope('model_%d' % i):
        hp = HyperParams()
        model = self._model_builder_fn(hp)

      with tf.variable_scope('loss_%d' % i):
        outputs = model.outputs
        output_names = keras_train_utils.generic_output_names(outputs)
        loss_functions = keras_train_utils.prepare_loss_functions(loss, output_names)
        target_tensors = model._process_target_tensor_for_compile(target_tensors)
        training_endpoints = []
        for o, n, l, t in zip(outputs, output_names,
                              loss_functions, target_tensors):
          endpoint = keras_train._TrainingEndpoint(o, n, l)
          endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
          training_endpoints.append(endpoint)

        opt = keras_optimizers.get(optimizer)
        _hyper = {}
        for n, v in opt._hyper.items():
          if isinstance(v, float):
            opt._set_hyper(n, hp.get_hparam(n, default_value=v))

        train_attrs[i] = {
            'model': model,
            'loss_functions': loss_functions,
            'target_tensors': target_tensors,
            'training_endpoints': training_endpoints,
            'optimizer': opt
        }
    self._train_attrs = train_attrs
    self._is_compiled = True

  def fit(x, y, ):

    if not self._is_compiled:
      raise ValueError("model is not compiled. Call compile() method first.")


