import tensorflow as tf
import numpy as np
import deep_tempering as dt
from keras.datasets import mnist
from keras.utils import np_utils


class AvgPoolWithWeights(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, activation=None, **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.activation = activation
        super(AvgPoolWithWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.strides,
                                                         padding=self.padding, data_format=self.data_format)
        w_init = tf.contrib.layers.xavier_initializer()

        self.w = self.add_weight(name='w',
                                      shape=(input_shape.as_list()[-1],),
                                      initializer=w_init,
                                      trainable=True, dtype=tf.float32)

        self.b = self.add_weight(name='b',
                                 shape=(input_shape.as_list()[-1],),
                                 initializer='zeros',
                                 trainable=True, dtype=tf.float32)

        super(AvgPoolWithWeights, self).build(input_shape)

    def call(self, x, **kwargs):
        x = self.avg_pool(x)
        x = x * self.w + self.b
        if self.activation:
            return self.activation(x)
        return x


    # def compute_output_shape(self, input_shape):
    #     return input_shape


class RBFEuclidean(tf.keras.layers.Layer):
    def __init__(self, units=10, activation=None, **kwargs):
        self.units = units
        self.activation = activation
        super(RBFEuclidean, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        w_init = tf.contrib.layers.xavier_initializer()
        self.w = self.add_weight(name='w',
                                 shape=(input_shape.as_list()[1], self.units),
                                 initializer=w_init,
                                 trainable=True, dtype=tf.float32)
        super(RBFEuclidean, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.reduce_sum(tf.square(x[..., None] - self.w), axis=1)
        if self.activation:
            return self.activation(x)
        return x


def lenet5_emnist_builder(hp):
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    inputs = tf.keras.layers.Input((32,32,1))
    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh',)(inputs)
    res = AvgPoolWithWeights(pool_size=(2, 2,), strides=(2, 2,), activation=tf.keras.activations.tanh)(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)
    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')(res)
    res = AvgPoolWithWeights(pool_size=(2, 2,), strides=(2, 2), activation=tf.keras.activations.tanh)(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh')(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Flatten()(res)
    res = tf.keras.layers.Dense(units=84, activation='tanh')(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = RBFEuclidean(units=26)(res)
    model = tf.keras.models.Model(inputs, res)
    return model


def lenet5_cifar10_builder(hp):
    dropout_rate = hp.get_hparam('dropout_rate', default_value=0.0)

    inputs = tf.keras.layers.Input((32,32,3))
    res = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu',)(inputs)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.MaxPooling2D(pool_size=(2, 2,), strides=(2, 2,))(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)


    res = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu')(res)
    res = tf.keras.layers.Flatten()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Dense(units=84, activation='relu')(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)

    res = tf.keras.layers.Dense(units=10, activation='softmax')(res) #ToDo: check whther softmax is present in research code or pass params to loss to work with logits
    model = tf.keras.models.Model(inputs, res)
    return model
