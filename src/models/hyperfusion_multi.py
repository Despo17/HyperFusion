import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Concatenate,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Embedding, Flatten
)
from tensorflow.keras.models import Model


def transformer_block(x, head_size=32, num_heads=2, ff_dim=64):
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x, x)
    x = LayerNormalization()(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)

    return LayerNormalization()(x + ff)


def build_multi_asset_hyperfusion(
    seq_len,
    num_features,
    num_assets
):

    # Sequence input
    seq_input = Input(shape=(seq_len, num_features))

    # Asset ID input
    asset_input = Input(shape=(1,))

    # Asset embedding
    asset_emb = Embedding(
        input_dim=num_assets,
        output_dim=8
    )(asset_input)

    asset_emb = Flatten()(asset_emb)

    # LSTM branch
    lstm_out = LSTM(64)(seq_input)

    # Transformer branch
    x = transformer_block(seq_input)
    x = transformer_block(x)
    trans_out = GlobalAveragePooling1D()(x)

    # Fuse sequence representations
    fused_seq = Concatenate()([lstm_out, trans_out])

    # Fuse asset embedding
    fused = Concatenate()([fused_seq, asset_emb])

    x = Dense(64, activation="relu")(fused)
    x = Dense(32, activation="relu")(x)

    output = Dense(1, activation="softplus")(x)

    model = Model(
        inputs=[seq_input, asset_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model