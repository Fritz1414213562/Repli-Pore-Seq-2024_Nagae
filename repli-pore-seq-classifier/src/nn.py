import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datareader import read_all
import numpy as np
import sys


def run(args):
    main(args.dataset, args.output)



def CNN_LSTM(shape, filters, kernels, res_cycle, n_hid_unit, mask_value, activation, l2_norm_coef, nseed, pool_size, do_rate):

    def ResidualBlock(x, filter, kernel_size, cycle_num):
        fx = layers.SeparableConv1D(filter, kernel_size, activation = activation, padding = "same", kernel_initializer = kernel_init, kernel_regularizer = keras.regularizers.l2(l2_norm_coef))(x)
        fx = layers.BatchNormalization()(fx)
        for icycle in range(1, cycle_num):
            fx = layers.Activation(activation)(fx)
            if icycle == cycle_num - 1:
                fx = layers.Dropout(do_rate)(fx)
            fx = layers.SeparableConv1D(filter, kernel_size, activation = activation, padding = "same", kernel_initializer = kernel_init, kernel_regularizer = keras.regularizers.l2(l2_norm_coef))(fx)
            fx = layers.BatchNormalization()(fx)
        out = layers.Add()([x, fx])
        #out = layers.Dropout(do_rate)(out)
        #out = layers.SpatialDropout1D(do_rate)(out)
        #out = layers.BatchNormalization()(out)
        return out

    if len(filters) != len(kernels):
        print("The number of filters is not consistent with that of kernels", file = sys.stderr)
        sys.exit()
    elif len(filters) < 1:
        print("The number of filters must be >= 1", file = sys.stderr)
        sys.exit()

    kernel_init = keras.initializers.he_normal(seed = nseed)

    inputs = layers.Input(shape = shape)
    x      = layers.Masking(mask_value = mask_value)(inputs)

    for filter, kernel in zip(filters, kernels):
        x  = layers.SeparableConv1D(filters = filter, kernel_size = kernel, kernel_regularizer = keras.regularizers.l2(l2_norm_coef), kernel_initializer = kernel_init, padding = "same")(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Activation(activation)(x)
        x  = ResidualBlock(x, filter, kernel, res_cycle)
        x  = layers.MaxPooling1D(pool_size = pool_size, padding = "same")(x)

    #x      = layers.LSTM(n_hid_unit, return_sequences = False)(x)
    x      = layers.LSTM(n_hid_unit, return_sequences = True)(x)
    x      = layers.Flatten()(x)
    x      = layers.Dense(1, kernel_regularizer = keras.regularizers.l2(l2_norm_coef), kernel_initializer = kernel_init)(x)
    x      = layers.Activation("sigmoid", trainable = False)(x)

    return models.Model(inputs = inputs, outputs = x)


def main(dataset_name, model_name):

    tf_seed = 81265925
    np_seed = 18879208
    in_seed = 76150417
    padding = -4096
    batch_size = 1024
    learning_rate = 0.001
    l2_norm_coef = 0.001
    do_rate = 0.00
    #loss_function = keras.losses.BinaryCrossentropy()
    loss_function = keras.losses.BinaryFocalCrossentropy()
    metrics = [keras.metrics.BinaryAccuracy()]
    nepochs = 1000
    
    # Variables
    history_name = model_name + "_learning_history.csv"
    
    filters = [64, 64, 64, 64, 64, 64, 64, 64]
    kernels = [16, 16, 16, 16, 16, 16, 16, 16]
    n_hid_unit = 128
    pool_size = 8
    cycle = 4
    #midl_activation = "tanh"
    activation = "relu"
    
    
    tf.random.set_seed(tf_seed)
    np.random.seed(np_seed)
    
    data = read_all(dataset_name)
    train_data = data["signal"]
    labels = np.reshape(data["label"], (train_data.shape[0], 1))
    shape = (train_data.shape[1], train_data.shape[2])
    
    model = CNN_LSTM(shape, filters, kernels, cycle, n_hid_unit, padding, activation, l2_norm_coef, in_seed, pool_size, do_rate)
    
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(loss = loss_function, optimizer = optimizer, metrics = metrics)
    early_stopping = EarlyStopping(monitor = "val_loss", mode = "auto", patience = 10, verbose = 2)
    learning_history = model.fit(train_data, labels, batch_size = batch_size, epochs = nepochs, validation_split = 0.1, callbacks = [early_stopping])
    models.save_model(model, filepath = model_name)
    
    with open(history_name, 'w') as ofs:
        history = learning_history.history
        n_steps = len(history["val_loss"])
        ofs.write("#{:>11s} {:>12s}\n".format("val_loss", "loss"))
        for istep in range(n_steps):
            ofs.write("{:12.6f} {:12.6f}\n".format(history["val_loss"][istep], history["loss"][istep]))


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", '-d', required = True)
    parser.add_argument("--output", '-o', required = True)
    args = parser.parse_args()
    
    main(args.dataset, args.output)
