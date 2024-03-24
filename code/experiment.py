import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import GRU, Dropout, Dense, Input, RepeatVector, Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilities import cyclic_coordinates, moving_average, prepare_covariance_matrix, sample_with_given_first_element, \
    compute_conditional_stats
from loss_functions import gaussian_loss, student_loss, cauchy_loss, correlated_gaussian_loss, gaussian_loss_15s
from models import GRUModel, TransformerModel
import joblib
from keras.models import load_model
import keras
import random
from kerastuner.tuners import Hyperband

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# model_name = "gru_gaussian"
# model_name = "transformer_gaussian"
# model_name = 'correlated_gaussian_rational_quadratic'
# model_name = 'correlated_gaussian_Exponentiated_Quadratic_GRU'
# model_name = 'cauchy'

#model_name = 'transformer_gaussian_15s'

# model_type = "GRU"
# model_type = "Transformer"
# model_type = "Correlated Gaussian" Standard: transformer
# model_type = "Correlated Gaussian GRU"

# loss_function_name = "gaussian"
# loss_function_name = "Correlated Gaussian"
# loss_function_name = "cauchy"
# loss_function_name = "transformed t"




loss_function_name = "gaussian"


if loss_function_name == "gaussian":
    if model_name == 'transformer_gaussian_15s':
        aggregation = 15
        N = 3600 // aggregation
        num_output_units = 2 * N
        loss_function = gaussian_loss_15s
    else:
        num_output_units = 7200
        loss_function = gaussian_loss
elif loss_function_name == "transformed t":
    num_output_units = 3600 * 3
    loss_function = student_loss
elif loss_function_name == "cauchy":
    num_output_units = 3600 * 2
    loss_function = cauchy_loss
elif loss_function_name == "Correlated Gaussian":
    # Forecast frequency 15s
    aggregation = 15
    N = 3600 // aggregation
    covariance_matrix = prepare_covariance_matrix(N, kernel_type='RationalQuadratic')
    loss_function = correlated_gaussian_loss(N, covariance_matrix)

# data import
day_ahead_features_train = pd.read_pickle(r"..\data\day_ahead_features_train.pkl")
day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")
frequency_train = pd.read_pickle(r"..\data\frequency_train.pkl")
frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")

day_ahead_features_train_np = day_ahead_features_train.to_numpy()
day_ahead_features_test_np = day_ahead_features_test.to_numpy()
frequency_train_np = frequency_train.to_numpy()
frequency_test_np = frequency_test.to_numpy()

inputs = day_ahead_features_train_np
inputs_test = day_ahead_features_test_np

scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)
# joblib.dump(scaler, 'scaler.gz')

inputs_test = scaler.transform(inputs_test)

outputs = frequency_train_np
outputs = outputs - 50.0  # Frequency Deviation
outputs_test = frequency_test_np - 50.0

if model_type == "Correlated Gaussian" or model_name == 'transformer_gaussian_15s' or model_type == "Correlated Gaussian GRU":

    outputs = outputs[:, ::aggregation]
    outputs_test = outputs_test[:, ::aggregation]

    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2,
                                                                            random_state=1)
else:
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2,
                                                                            random_state=1)

input_shape = (14,)
input_layer = Input(shape=input_shape)

# Create and compile the model

if model_type == "GRU":
    # Model specifications
    num_gru_layers = 1
    num_dense_layers = 6
    include_dropout = True
    dropout_rate = 0.1
    gru_units = 128
    dense_units = 128
    output_units = num_output_units

    model = GRUModel(input_shape, num_gru_layers, num_dense_layers, include_dropout, dropout_rate, gru_units,
                     dense_units, output_units)
    model = model.build_model()
    model.compile(optimizer='adam', loss=loss_function)

elif model_type == "Transformer":
    key_dim = 32  # 16
    dense_config = {'units_per_layer': 128, 'num_layers': 6}
    output_units = num_output_units
    num_heads = 4
    num_attention_blocks = 1
    num_dense_layers_after_attention = 6
    include_dropout = True
    model = TransformerModel(
        input_shape=input_shape,
        key_dim=key_dim,
        dense_config=dense_config,
        output_units=output_units,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        num_dense_layers_after_attention=num_dense_layers_after_attention,
        include_dropout=include_dropout
    )

    model = model.build_model()
    model.compile(optimizer='adam', loss=loss_function)

elif model_type == "Correlated Gaussian":

    key_dim = 32
    dense_config = {'units_per_layer': 128, 'num_layers': 6}
    output_units = N + 1
    num_heads = 4
    num_attention_blocks = 1
    num_dense_layers_after_attention = 6
    include_dropout = True
    model = TransformerModel(
        input_shape=input_shape,
        key_dim=key_dim,
        dense_config=dense_config,
        output_units=output_units,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        num_dense_layers_after_attention=num_dense_layers_after_attention,
        include_dropout=include_dropout
    )

    model = model.build_model()
    model.compile(optimizer='adam', loss=loss_function)


elif model_type == "Correlated Gaussian GRU":

    output_units = N + 1
    num_gru_layers = 1
    num_dense_layers = 6
    include_dropout = True
    dropout_rate = 0.1
    gru_units = 128
    dense_units = 128
    model = GRUModel(input_shape, num_gru_layers, num_dense_layers, include_dropout, dropout_rate, gru_units,
                     dense_units, output_units)
    model = model.build_model()
    model.compile(optimizer='adam', loss=loss_function)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.000001)

# Model Training
history = model.fit(train_inputs, train_outputs, epochs=100, validation_data=(val_inputs, val_outputs), batch_size=128,
                    callbacks=[early_stopping, lr_reduction])

# Model
# model.save(f"../trained_models/{model_name}")


