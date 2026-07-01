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
    Reshape,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import AdamW


class FeaturePositionalEmbedding(tf.keras.layers.Layer):
    """Learnable per-feature position embedding, added to the projected tokens."""

    def __init__(self, num_features: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.num_features, self.projection_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="feature_pos_embedding",
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "projection_dim": self.projection_dim})
        return config


def compile_classifier(
    model: Model,
    learning_rate: float = 3e-4,
    weight_decay: float = 3e-5,
    clipnorm: float = 1.0,
    label_smoothing: float = 0.01,
) -> Model:
    """Compile the proposed model with stable optimizer settings."""

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=optimizer,
        loss=loss,
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


def multi_scale_feature_mixer(x, projection_dim: int, dropout_rate: float, l2, name_prefix: str):
    """Capture short local feature interactions at multiple receptive fields."""

    conv3 = Conv1D(
        projection_dim // 3,
        kernel_size=3,
        padding="same",
        activation="gelu",
        kernel_regularizer=l2,
        name=f"{name_prefix}_conv3",
    )(x)
    conv5 = Conv1D(
        projection_dim // 3,
        kernel_size=5,
        padding="same",
        activation="gelu",
        kernel_regularizer=l2,
        name=f"{name_prefix}_conv5",
    )(x)
    conv7 = Conv1D(
        projection_dim - 2 * (projection_dim // 3),
        kernel_size=7,
        padding="same",
        activation="gelu",
        kernel_regularizer=l2,
        name=f"{name_prefix}_conv7",
    )(x)
    mixed = Concatenate(name=f"{name_prefix}_concat")([conv3, conv5, conv7])
    mixed = Dropout(dropout_rate, name=f"{name_prefix}_dropout")(mixed)
    return Add(name=f"{name_prefix}_residual")([x, mixed])


def squeeze_excitation_block(x, ratio: int = 8, name: str = "se"):
    """Channel attention to emphasize informative sequence states."""

    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
    excite = Dense(max(channels // ratio, 8), activation="gelu", name=f"{name}_reduce")(squeeze)
    excite = Dense(channels, activation="sigmoid", name=f"{name}_expand")(excite)
    excite = Reshape((1, channels), name=f"{name}_expand_dims")(excite)
    return Multiply(name=f"{name}_scale")([x, excite])


def build_transformer_lstm(
    input_shape: tuple[int, int],
    num_classes: int,
    projection_dim: int = 192,
    lstm_units: int = 160,
    attention_heads: int = 8,
    transformer_blocks: int = 3,
    lstm_layers: int = 2,
    attention_dropout: float = 0.10,
    dropout_rate: float = 0.14,
    ffn_units: int = 512,
    dense_units: int = 384,
    learning_rate: float = 3e-4,
    weight_decay: float = 3e-5,
    label_smoothing: float = 0.01,
    clipnorm: float = 1.0,
    gaussian_noise: float = 0.0,
    use_leaky_relu_head: bool = False,
) -> Model:
    """Build a high-capacity Transformer-Enhanced LSTM for 96%+ accuracy targets.

    Improvements over the baseline architecture:
    - grouped multivariate feature timesteps
    - multi-scale convolutional feature mixing
    - stacked bidirectional LSTM encoders
    - deeper Transformer encoder stack
    - squeeze-and-excitation channel attention
    - dual pooling with a wider classification head
    """

    if projection_dim % attention_heads != 0:
        raise ValueError("projection_dim must be divisible by attention_heads.")

    l2 = regularizers.l2(weight_decay)
    key_dim = projection_dim // attention_heads

    inputs = Input(shape=input_shape, name="network_features")

    x = GaussianNoise(gaussian_noise, name="input_gaussian_noise")(inputs) if gaussian_noise > 0 else inputs
    x = Dense(projection_dim, kernel_regularizer=l2, name="feature_projection")(x)
    x = FeaturePositionalEmbedding(input_shape[0], projection_dim, name="feature_pos_embed")(x)
    x = LayerNormalization(epsilon=1e-6, name="projection_norm")(x)
    x = SpatialDropout1D(dropout_rate * 0.5, name="projection_dropout")(x)

    x = multi_scale_feature_mixer(
        x,
        projection_dim=projection_dim,
        dropout_rate=dropout_rate * 0.5,
        l2=l2,
        name_prefix="feature_mixer",
    )
    x = LayerNormalization(epsilon=1e-6, name="feature_mixer_norm")(x)

    for layer_id in range(1, lstm_layers + 1):
        return_sequences = layer_id < lstm_layers or transformer_blocks > 0
        x = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=dropout_rate * 0.45,
                recurrent_dropout=0.0,
                kernel_regularizer=l2,
                recurrent_regularizer=l2,
                name=f"lstm_encoder_{layer_id}",
            ),
            name=f"bidirectional_lstm_{layer_id}",
        )(x)
        if return_sequences:
            x = Dense(projection_dim, kernel_regularizer=l2, name=f"lstm_projection_{layer_id}")(x)
            x = Dropout(dropout_rate * 0.75, name=f"lstm_projection_dropout_{layer_id}")(x)

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

    x = squeeze_excitation_block(x, ratio=8, name="sequence_se")

    avg_pool = GlobalAveragePooling1D(name="global_average_pooling")(x)
    max_pool = GlobalMaxPooling1D(name="global_max_pooling")(x)
    pooled = Concatenate(name="pooled_features")([avg_pool, max_pool])

    x = Dense(dense_units, kernel_regularizer=l2, name="classifier_dense")(pooled)
    x = LeakyReLU(negative_slope=0.05, name="classifier_dense_leaky_relu")(x) if use_leaky_relu_head else tf.keras.activations.gelu(x)
    x = Dropout(dropout_rate, name="classifier_dropout")(x)
    x = Dense(dense_units // 2, kernel_regularizer=l2, name="classifier_refine")(x)
    x = LeakyReLU(negative_slope=0.05, name="classifier_refine_leaky_relu")(x) if use_leaky_relu_head else tf.keras.activations.gelu(x)
    x = Dropout(dropout_rate * 0.5, name="classifier_refine_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return compile_classifier(
        Model(inputs, outputs, name="Transformer_Enhanced_LSTM"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        clipnorm=clipnorm,
    )


def set_global_determinism(seed: int = 42) -> None:
    """Set random seeds so experiments are easier to reproduce."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some TensorFlow builds do not expose deterministic kernels on Windows.
        pass
