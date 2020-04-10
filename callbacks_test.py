import pt_ensemble
import callbacks as cbks
from pt_ensemble_test import model_builder

def configure_callbacks_test():
  model = pt_ensemble.EnsembleModel(model_builder)
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
      'verbose': 1
  }
  callbacklist = cbks.configure_callbacks([], model, **kwargs)
  kwargs.update({
      'metrics': model.metrics_names + ['val_' + m for m in model.metrics_names]
  })

  kwargs['steps'] = (kwargs['steps_per_epoch'], kwargs.pop('steps_per_epoch'))[0]

  # test that params are stored as intended
  assert kwargs == callbacklist.params