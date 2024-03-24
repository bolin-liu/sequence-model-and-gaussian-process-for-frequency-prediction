from keras.layers import GRU, RepeatVector
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, LayerNormalization, MultiHeadAttention, Flatten, Reshape
from keras.models import Model


def dense_layers(input_layer, num_hidden_layers, num_neurons, output_dim):
    """
    Creates a nn with a specific number of hidden layers and neurons.

    Args:
    input_layer: The input layer or the preceding tensor.
    num_hidden_layers: The number of hidden layers.
    num_neurons: The number of neurons in each hidden layer.
    output_dim: The dimension of the output layer.

    Returns:
    The output tensor
    """
    x = input_layer
    for _ in range(num_hidden_layers):
        x = Dense(num_neurons, activation='relu')(x)
    output_layer = Dense(output_dim, activation='linear')(x)
    return output_layer


class GRUModel:
    def __init__(self, input_shape, num_gru_layers, num_dense_layers, include_dropout, dropout_rate, gru_units,
                 dense_units, output_units):
        self.input_shape = input_shape
        self.num_gru_layers = num_gru_layers
        self.num_dense_layers = num_dense_layers
        self.include_dropout = include_dropout
        self.dropout_rate = dropout_rate
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.output_units = output_units

    def create_gru_branch(self):
        input_layer = Input(shape=self.input_shape)
        x = RepeatVector(180)(input_layer)

        # GRU layers
        for i in range(self.num_gru_layers):
            return_sequences = i < self.num_gru_layers - 1 
            x = GRU(self.gru_units, return_sequences=return_sequences)(x)
            if self.include_dropout:
                x = Dropout(self.dropout_rate)(x)

        for _ in range(self.num_dense_layers):
            x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(0.1)(x)
        # Output layer
        output_layer = Dense(self.output_units, activation=None)(x)
        return input_layer, output_layer

    def build_model(self):
        input_layer, output_layer = self.create_gru_branch()
        model = Model(inputs=input_layer, outputs=output_layer)

        return model


class TransformerModel:
    def __init__(self, input_shape, key_dim=64, dense_config=None, output_units=7200, num_heads=12,
                 num_attention_blocks=1, num_dense_layers_after_attention=4, include_dropout=True):
        self.input_shape = input_shape
        self.key_dim = key_dim
        self.dense_config = dense_config or {'units_per_layer': 128, 'num_layers': 4, 'dropout_rate': 0.2}
        self.output_units = output_units
        self.num_heads = num_heads
        self.num_attention_blocks = num_attention_blocks
        self.num_dense_layers_after_attention = num_dense_layers_after_attention
        self.include_dropout = include_dropout

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # Initial dense transformations
        x = Dense(self.key_dim * 14, activation='relu')(input_layer)
        x = Dense(self.key_dim * 14, activation='relu')(x)
        x = Dense(self.key_dim * 14, activation='linear')(x)
        x = Reshape((14, self.key_dim))(x)

        # Attention blocks
        for _ in range(self.num_attention_blocks):
            
            keys = dense_layers(x, 4, 64, self.key_dim)
            values = dense_layers(x, 4, 64, self.key_dim)
            queries = dense_layers(x, 4, 64, self.key_dim)

            attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(queries, values, keys)
            x = Add()([x, attention_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)

        for _ in range(self.num_dense_layers_after_attention):
            x = Dense(self.dense_config['units_per_layer'], activation='relu')(x)
            
        x = Dropout(0.05)(x)
        output_layer = Dense(self.output_units, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
