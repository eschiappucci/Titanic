def tf_regression(mat, output_shape = 1, rate=1e-5, loss_func='mse', metric='mean_absolute_percentage_error'):
    model = keras.Sequential([
        keras.layers.ZeroPadding1D(2, input_shape=mat[0].shape),
        keras.layers.LocallyConnected1D(128, 4, data_format='channels_last'),
        keras.layers.LocallyConnected1D(64, 5, data_format='channels_last'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(output_shape, activation='exponential')
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

def recompile(loc, rate=1e-5, loss_func='mse', metric='mean_absolute_percentage_error'):
    model = tf.keras.models.load_model(loc, compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
    return model
