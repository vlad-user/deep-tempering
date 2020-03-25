import tensorflow as tf
from tensorflow.python.keras.engine import training_utils as keras_train_utils
import numpy as np

import pt_train_utils as train_utils

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
    self._exchange_hparams = None
    self._train_attrs = None

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

      opt = tf.keras.optimizers.get(optimizer)
      _hyper = {}
      for n, v in opt._hyper.items():
        if isinstance(v, float):
          opt._set_hyper(n, hp.get_hparam(n, default_value=v))

      train_attrs[i] = {
          'model': model,
          'loss_functions': loss_functions,
          'optimizer': opt
      }
    self._train_attrs = train_attrs
    self._is_compiled = True

  def fit(self, x, y):
    if not self._is_compiled:
      raise ValueError("model is not compiled. Call compile() method first.")

    # create tensors for true labels
    # the same tensor is fed to all ensemble losses
    target_tensor = train_utils.create_training_target(
        train_utils.infer_shape_from_numpy_array(y))
    self._target_tensor = target_tensor

    # create losses and optimization step operation
    for i in range(self.n_replicas):
      loss_function = self._train_attrs[i]['loss_function']
      y_pred = self._train_attrs[i]['model'].outputs[0]
      loss_function = self._train_attrs[i]['loss_functions'][0]
      loss = loss_function(target_tensor, y_pred)
      self._train_attrs[i]['loss'] = loss
      var_list = self._train_attrs[i]['model'].trainable_variables

      train_op = self._train_attrs[i]['optimizer'].minimize(loss,
                                                            var_list)
      self._train_attrs[i]['train_op'] = train_op

    return self._train_attrs
