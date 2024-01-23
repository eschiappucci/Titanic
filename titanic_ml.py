import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def tf_regression(mat, output_shape = 1, rate=1e-5, loss_func='binary_crossentropy', metric='binary_accuracy'):
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=mat[0].shape),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(output_shape, activation='sigmoid')
        ])
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
    return model

def plot_metric(history, metric='loss'):
    plt.plot(history.history[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Change of '+metric)
    plt.legend()
    plt.show()

def recompile(loc, rate=1e-5, loss_func='binary_crossentropy', metric='binary_accuracy'):
    model = tf.keras.models.load_model(loc, compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
    return model
