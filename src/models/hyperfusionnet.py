import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, GlobalAveragePooling1D,
    Concatenate, LayerNormalization, MultiHeadAttention, Dropout
)
from tensorflow.keras.models import Model


def transformer_block(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    """
    Transformer encoder block
    """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_hyperfusionnet(seq_len, n_features):
    """
    Hybrid LSTM + Transformer model
    """
    inp = Input(shape=(seq_len, n_features))

    # LSTM branch (temporal memory)
    lstm = LSTM(64)(inp)
    lstm = Dense(32, activation='relu')(lstm)

    # Transformer branch (global attention)
    x = transformer_block(inp)
    x = transformer_block(x)
    trans = GlobalAveragePooling1D()(x)
    trans = Dense(32, activation='relu')(trans)

    # Fusion
    fusion = Concatenate()([lstm, trans])
    fusion = Dense(32, activation='relu')(fusion)
    out = Dense(1,activation="softplus")(fusion)

    model = Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model