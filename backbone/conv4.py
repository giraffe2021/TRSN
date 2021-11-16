import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, Sequential, optimizers, metrics


def conv4_net(input_shape=None, pooling=None):
    if input_shape is None:
        input_shape = (84, 84, 3)
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same',
                               input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if pooling == 'avg':
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif pooling == 'max':
        model.add(tf.keras.layers.GlobalMaxPooling2D())
    x = tf.keras.layers.Input(shape=input_shape)
    out = model(x)
    return tf.keras.Model(x, out)
