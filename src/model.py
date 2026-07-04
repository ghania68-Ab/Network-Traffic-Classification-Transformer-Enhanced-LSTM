"""Proposed Transformer-Enhanced LSTM model for network traffic classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Add,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GaussianNoise,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    LayerNormalization,
    LeakyReLU,
    Multiply,
    MultiHeadAttention,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import AdamW


def compile_classifier(
    model: Model,
    learning_rate: float = 5e-4,
    weight_decay: float = 5e-5,
    clipnorm: float = 1.0,
) -> Model:
    """Compile the proposed model with stable optimizer settings."""

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def transformer_encoder_block(
    x,
    block_id: int,
    projection_dim: int,
    attention_heads: int,
    key_dim: int,
    ffn_units: int,
    dropout_rate: float,
    attention_dropout: float,
    l2,
):
    """Pre-norm Transformer encoder block with attention and FFN residuals."""

    attention_input = LayerNormalization(epsilon=1e-6, name=f"attention_{block_id}_input_norm")(x)
    attention_output = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=key_dim,
        value_dim=key_dim,
        dropout=attention_dropout,
        output_shape=projection_dim,
        name=f"multi_head_self_attention_{block_id}",
    )(attention_input, attention_input)
    attention_output = Dropout(dropout_rate, name=f"attention_{block_id}_dropout")(attention_output)
    x = Add(name=f"attention_{block_id}_residual")([x, attention_output])

    ffn_input = LayerNormalization(epsilon=1e-6, name=f"ffn_{block_id}_input_norm")(x)
    ffn = Dense(ffn_units, activation="gelu", kernel_regularizer=l2, name=f"ffn_{block_id}_expand")(ffn_input)
    ffn = Dropout(dropout_rate, name=f"ffn_{block_id}_dropout")(ffn)
    ffn = Dense(projection_dim, kernel_regularizer=l2, name=f"ffn_{block_id}_project")(ffn)
    x = Add(name=f"ffn_{block_id}_residual")([x, ffn])
    return LayerNormalization(epsilon=1e-6, name=f"encoder_{block_id}_output_norm")(x)


def build_transformer_lstm(
    input_shape: tuple[int, int],
    num_classes: int,
    projection_dim: int = 128,
    lstm_units: int = 128,
    attention_heads: int = 4,
    transformer_blocks: int = 2,
    attention_dropout: float = 0.12,
    dropout_rate: float = 0.18,
    ffn_units: int = 384,
    dense_units: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 5e-5,
) -> Model:
    """Build an accuracy-focused but regularized Transformer-Enhanced LSTM.

    The model keeps the required LSTM + attention methodology, but improves
    feature learning with a wider projection, lightweight convolutional feature
    mixer, bidirectional LSTM encoder, stacked Transformer blocks, and dual
    pooling. These changes increase capacity while dropout, AdamW, L2, and
    gradient clipping control overfitting.
    """

    if projection_dim % attention_heads != 0:
        raise ValueError("projection_dim must be divisible by attention_heads.")

    l2 = regularizers.l2(weight_decay)
    key_dim = projection_dim // attention_heads

    inputs = Input(shape=input_shape, name="network_features")

    # Learn a richer embedding for each normalized tabular feature timestep.
    x = Dense(projection_dim, kernel_regularizer=l2, name="feature_projection")(inputs)
    x = LayerNormalization(epsilon=1e-6, name="projection_norm")(x)
    x = SpatialDropout1D(dropout_rate * 0.5, name="projection_dropout")(x)

    # Local feature mixer helps the model combine neighboring engineered flow
    # features before the recurrent encoder, while the residual preserves the
    # original projection signal.
    conv = Conv1D(
        projection_dim,
        kernel_size=3,
        padding="same",
        activation="gelu",
        kernel_regularizer=l2,
        name="feature_mixer_conv",
    )(x)
    conv = Dropout(dropout_rate * 0.5, name="feature_mixer_dropout")(conv)
    x = Add(name="feature_mixer_residual")([x, conv])
    x = LayerNormalization(epsilon=1e-6, name="feature_mixer_norm")(x)

    # Bidirectional context is justified because feature order is non-causal.
    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate * 0.5,
            kernel_regularizer=l2,
            recurrent_regularizer=l2,
            name="lstm_encoder",
        ),
        name="bidirectional_lstm",
    )(x)
    x = Dense(projection_dim, kernel_regularizer=l2, name="lstm_projection")(x)
    x = Dropout(dropout_rate, name="lstm_projection_dropout")(x)

    for block_id in range(1, transformer_blocks + 1):
        x = transformer_encoder_block(
            x=x,
            block_id=block_id,
            projection_dim=projection_dim,
            attention_heads=attention_heads,
            key_dim=key_dim,
            ffn_units=ffn_units,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            l2=l2,
        )

    avg_pool = GlobalAveragePooling1D(name="global_average_pooling")(x)
    max_pool = GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = Concatenate(name="pooled_features")([avg_pool, max_pool])

    x = Dense(dense_units, activation="gelu", kernel_regularizer=l2, name="classifier_dense")(x)
    x = Dropout(dropout_rate, name="classifier_dropout")(x)
    x = Dense(dense_units // 2, activation="gelu", kernel_regularizer=l2, name="classifier_refine")(x)
    x = Dropout(dropout_rate * 0.5, name="classifier_refine_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return compile_classifier(
        Model(inputs, outputs, name="Transformer_Enhanced_LSTM"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def set_global_determinism(seed: int = 42) -> None:
    """Set random seeds so experiments are easier to reproduce."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some TensorFlow builds do not expose deterministic kernels on Windows.
        pass