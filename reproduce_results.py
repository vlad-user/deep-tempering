import tensorflow as tf

# tf.enable_eager_execution()
import numpy as np
# import tensorflow_datasets as tfds
import deep_tempering as dt
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset

import wandb
from wandb.keras import WandbCallback



wandb.init(
  project="deep-tempering",
  name="test-cifar10-accuracy",
  notes="",
  config={
    "model_name": "lenet5",
    "dataset_name": "cifar10",
    "n_replicas": 2,
    "swap_step": 600,
    "burn_in": 10000,
    "batch_size": 128,
    "epochs": 400,
    "lr_range": [0.01, 0.01],
    "dropout_range": [0.45, 0.45]
}
)
config = wandb.config


model_builders = {'lenet5': {'cifar10': lenet5_cifar10_builder, 'emnist': lenet5_emnist_builder}}


def prepare_data(ds):
    if ds == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif ds == 'emnist':
        x_train, y_train, x_test, y_test = get_emnist_letters()

        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), #ToDo: add augmentations
           mode='constant', constant_values=0)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                         mode='constant', constant_values=0)
        y_train = np.int32(y_train) - 1
        y_test = np.int32(y_test) - 1

    x_train = np.float32(x_train) / 255.
    x_test = np.float32(x_test) / 255.


    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    return x_train, y_train, x_val, y_val, x_test, y_test,


x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(config.dataset_name)

model = dt.EnsembleModel(model_builders[config.model_name][config.dataset_name])

hp = {
    'learning_rate': np.linspace(config.lr_range[0], config.lr_range[0], config.n_replicas),
    # 'dropout_rate': np.linspace(config.dropout_range[0], config.dropout_range[0], config.n_replicas)
}

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              n_replicas=config.n_replicas)


history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    hyper_params=hp,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    swap_step=config.swap_step,
                    burn_in=config.burn_in,)
                    # callbacks=[WandbCallback(data_type="image",)]) wandbcallback doesnt work with EnsembleModel, it lacks several attributes


for step in range(len(history.history['acc_0'])):
    wandb.log({k: history.history[k][step] for k in sorted(history.history.keys())}, step=step)



# access the optimal (not compiled) keras' model instance
optimal_model = model.optimal_model()

# inference only on the trained optimal model
predicted = optimal_model.predict(x_test)