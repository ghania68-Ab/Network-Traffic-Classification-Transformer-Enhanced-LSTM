"""Dataset-aware hyperparameter presets for the proposed model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingConfig:
    """Accuracy-focused training preset for one dataset."""

    projection_dim: int = 192
    lstm_units: int = 160
    attention_heads: int = 8
    transformer_blocks: int = 3
    lstm_layers: int = 2
    attention_dropout: float = 0.10
    dropout_rate: float = 0.14
    ffn_units: int = 512
    dense_units: int = 384
    learning_rate: float = 3e-4
    weight_decay: float = 3e-5
    label_smoothing: float = 0.0
    clipnorm: float = 1.0
    epochs: int = 100
    batch_size: int = 256
    warmup_epochs: int = 8
    min_epochs_before_stopping: int = 60
    early_stopping_patience: int = 18
    mixup_alpha: float = 0.0
    feature_group_size: int = 1
    oversample_ratio: float = 0.85
    correlation_threshold: float = 0.995
    variance_threshold: float = 1e-6
    class_weight_mode: str = "sqrt"
    gaussian_noise: float = 0.0
    use_leaky_relu_head: bool = False


DATASET_CONFIGS: Dict[str, TrainingConfig] = {
    "cic": TrainingConfig(
        projection_dim=224,
        lstm_units=192,
        attention_heads=8,
        transformer_blocks=4,
        lstm_layers=2,
        attention_dropout=0.08,
        dropout_rate=0.12,
        ffn_units=640,
        dense_units=512,
        learning_rate=2.5e-4,
        weight_decay=2e-5,
        label_smoothing=0.0,
        epochs=50,
        batch_size=192,
        warmup_epochs=10,
        min_epochs_before_stopping=50,
        early_stopping_patience=40,
        mixup_alpha=0.0,
        feature_group_size=1,
        oversample_ratio=0.0,
        correlation_threshold=0.995,
        class_weight_mode="none",
        gaussian_noise=0.003,
        use_leaky_relu_head=True,
    ),
    "unsw": TrainingConfig(
        projection_dim=192,
        lstm_units=160,
        attention_heads=8,
        transformer_blocks=3,
        lstm_layers=2,
        attention_dropout=0.10,
        dropout_rate=0.16,
        ffn_units=512,
        dense_units=384,
        learning_rate=2.2e-4,
        weight_decay=4e-5,
        label_smoothing=0.01,
        epochs=50,
        batch_size=128,
        warmup_epochs=8,
        min_epochs_before_stopping=50,
        early_stopping_patience=32,
        mixup_alpha=0.0,
        feature_group_size=1,
        oversample_ratio=0.0,
        correlation_threshold=0.98,
        class_weight_mode="none",
        gaussian_noise=0.002,
        use_leaky_relu_head=True,
    ),
}


def get_training_config(dataset: str) -> TrainingConfig:
    """Return the tuned preset for a supported dataset key."""

    key = dataset.lower()
    if key not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported values: {supported}.")
    return DATASET_CONFIGS[key]
