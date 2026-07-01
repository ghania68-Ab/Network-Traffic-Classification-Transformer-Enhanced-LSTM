"""Train and evaluate the proposed Transformer-Enhanced LSTM model.

Run examples:
    python src/train.py --dataset cic
    python src/train.py --dataset unsw
    python src/train.py --all-datasets
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

from config import TrainingConfig, get_training_config
from evaluate import (
    evaluate_model,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_dataset_comparison,
    plot_loss_curve,
    save_metrics,
)
from model import build_transformer_lstm, set_global_determinism
from preprocessing import prepare_dataset

<<<<<<< HEAD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


def cosine_warmup_schedule(epoch: int, lr: float, config: TrainingConfig) -> float:
    """Warmup followed by cosine decay for stable late-stage fine-tuning."""

    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


=======
>>>>>>> e38505b (Update project with latest changes)
MODEL_NAME = "Transformer-Enhanced LSTM"
DATASET_SLUGS = {"cic": "cic_darknet2020", "unsw": "unsw_nb15"}


<<<<<<< HEAD
class RestoreBestValidationWeights(Callback):
    """Keep the best validation-accuracy weights without stopping training early."""

    def __init__(self, monitor: str = "val_accuracy"):
        super().__init__()
        self.monitor = monitor
        self.best = -float("inf")
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is not None and current > self.best:
            self.best = float(current)
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored best validation weights from epoch {self.best_epoch} ({self.monitor}: {self.best:.4f}).")


def build_callbacks(
    results_dir: Path,
    dataset_slug: str,
    config: TrainingConfig,
    save_model: bool = False,
):
    """Return callbacks tuned for best validation accuracy."""

    callbacks = [
        RestoreBestValidationWeights(monitor="val_accuracy"),
        LearningRateScheduler(
            lambda epoch, lr: cosine_warmup_schedule(epoch, lr, config),
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(4, config.early_stopping_patience // 3),
=======
def build_callbacks(results_dir: Path, dataset_slug: str, save_model: bool = False):
    """Return callbacks tuned for best validation accuracy within 50 epochs."""

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
>>>>>>> e38505b (Update project with latest changes)
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    if save_model:
        results_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                results_dir / f"{dataset_slug}_transformer_lstm_best.keras",
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        )
    return callbacks


<<<<<<< HEAD
def resolve_class_weight_mode(dataset: str, requested_mode: str, config: TrainingConfig) -> str:
    """Choose a class-weight strategy without touching the test set."""

    if requested_mode != "auto":
        return requested_mode
    return config.class_weight_mode
=======
def resolve_class_weight_mode(dataset: str, requested_mode: str) -> str:
    """Choose a default class-weight strategy without touching the test set.

    UNSW-NB15 official test accuracy is often reduced by aggressive class
    weighting because the model predicts more rare attacks. For an
    accuracy-focused run, no class weights is the most appropriate default on
    UNSW. CIC keeps mild square-root weights because the main dataset benefits
    from supporting rare Tor/VPN classes without extreme over-correction.
    """

    if requested_mode != "auto":
        return requested_mode
    return "none" if dataset == "unsw" else "sqrt"
>>>>>>> e38505b (Update project with latest changes)


def compute_weights(y_train: np.ndarray, mode: str = "sqrt") -> Dict[int, float] | None:
    """Return class weights for imbalanced traffic datasets."""

    if mode == "none":
        return None
    classes = np.unique(y_train)
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = balanced if mode == "balanced" else np.sqrt(balanced)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


<<<<<<< HEAD
def build_model(input_shape: tuple[int, int], num_classes: int, config: TrainingConfig):
    """Build the proposed model using dataset-aware tuned defaults."""
=======
def build_model(input_shape: tuple[int, int], num_classes: int):
    """Build the single proposed model with accuracy-focused tuned defaults."""
>>>>>>> e38505b (Update project with latest changes)

    return build_transformer_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
<<<<<<< HEAD
        projection_dim=config.projection_dim,
        lstm_units=config.lstm_units,
        attention_heads=config.attention_heads,
        transformer_blocks=config.transformer_blocks,
        lstm_layers=config.lstm_layers,
        attention_dropout=config.attention_dropout,
        dropout_rate=config.dropout_rate,
        ffn_units=config.ffn_units,
        dense_units=config.dense_units,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        clipnorm=config.clipnorm,
        gaussian_noise=config.gaussian_noise,
        use_leaky_relu_head=config.use_leaky_relu_head,
    )


def make_mixup_dataset(
    X: np.ndarray,
    y: np.ndarray,
=======
        projection_dim=128,
        lstm_units=128,
        attention_heads=4,
        transformer_blocks=2,
        attention_dropout=0.12,
        dropout_rate=0.18,
        ffn_units=384,
        dense_units=256,
        learning_rate=5e-4,
        weight_decay=5e-5,
    )


def train_model(
    data,
    dataset_slug: str,
    epochs: int,
>>>>>>> e38505b (Update project with latest changes)
    batch_size: int,
    num_classes: int,
    mixup_alpha: float,
) -> tf.data.Dataset:
    """Create a shuffled, prefetched dataset with batch-level mixup augmentation."""

    labels = tf.one_hot(y, num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((X, labels))
    dataset = dataset.shuffle(buffer_size=min(len(X), 8192), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    if mixup_alpha <= 0:
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _mixup(batch_x, batch_y):
        batch_size_tensor = tf.shape(batch_x)[0]
        lam = tf.random.uniform([], 1.0 - mixup_alpha, 1.0)
        indices = tf.random.shuffle(tf.range(batch_size_tensor))
        mixed_x = lam * batch_x + (1.0 - lam) * tf.gather(batch_x, indices)
        mixed_y = lam * batch_y + (1.0 - lam) * tf.gather(batch_y, indices)
        return mixed_x, mixed_y

    return dataset.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def train_model(
    data,
    dataset_slug: str,
    config: TrainingConfig,
    results_dir: Path,
    class_weight_mode: str,
    save_model: bool,
) -> Tuple[object, object, Dict[str, object]]:
    """Train the proposed model and evaluate it on the held-out test split."""

<<<<<<< HEAD
    model = build_model(data.input_shape, data.num_classes, config)
=======
    model = build_model(data.input_shape, data.num_classes)
>>>>>>> e38505b (Update project with latest changes)
    weights = compute_weights(data.y_train, class_weight_mode)
    print("\n" + "=" * 72)
    print(f"Dataset            : {data.dataset_name}")
    print(f"Model              : {MODEL_NAME}")
<<<<<<< HEAD
    print(f"Input shape        : {data.input_shape}")
    print(f"Epochs             : {config.epochs}")
    print("Early stopping     : disabled; full epoch schedule will run")
    print(f"Batch              : {config.batch_size}")
    print(f"Mixup alpha        : {config.mixup_alpha}")
=======
    print(f"Epochs             : {epochs}")
    print(f"Batch              : {batch_size}")
>>>>>>> e38505b (Update project with latest changes)
    print(f"Class weight mode  : {class_weight_mode}")
    print("=" * 72 + "\n")

    y_val_oh = tf.one_hot(data.y_val, data.num_classes)
    train_ds = make_mixup_dataset(
        data.X_train,
        data.y_train,
<<<<<<< HEAD
        batch_size=config.batch_size,
        num_classes=data.num_classes,
        mixup_alpha=config.mixup_alpha,
    )
    val_ds = tf.data.Dataset.from_tensor_slices((data.X_val, y_val_oh)).batch(config.batch_size).prefetch(
        tf.data.AUTOTUNE
    )

    steps_per_epoch = math.ceil(len(data.X_train) / config.batch_size)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=build_callbacks(results_dir, dataset_slug, config=config, save_model=save_model),
=======
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(results_dir, dataset_slug, save_model=save_model),
>>>>>>> e38505b (Update project with latest changes)
        class_weight=weights,
        verbose=1,
    )

    print("\n" + "=" * 72)
    print(f"Finished Training : {MODEL_NAME}")
    print("=" * 72 + "\n")

    return model, history, evaluate_model(model, data.X_test, data.y_test, data.class_names)


def run_experiment(
    dataset: str = "cic",
    label_column: str | None = None,
    data_dir: str | Path = "dataset",
    results_dir: str | Path = "results",
<<<<<<< HEAD
    epochs: int | None = None,
    batch_size: int | None = None,
=======
    epochs: int = 50,
    batch_size: int = 128,
>>>>>>> e38505b (Update project with latest changes)
    max_samples: int | None = None,
    class_weight_mode: str = "auto",
    save_model: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Run the proposed model on one dataset and return experiment details."""

    set_global_determinism(seed)
<<<<<<< HEAD
    config = get_training_config(dataset)
    if epochs is not None:
        config = TrainingConfig(**{**config.__dict__, "epochs": epochs})
    if batch_size is not None:
        config = TrainingConfig(**{**config.__dict__, "batch_size": batch_size})

    data = prepare_dataset(
        dataset=dataset,
        data_dir=data_dir,
        label_column=label_column,
        max_samples=max_samples,
        training_config=config,
    )
    dataset_slug = DATASET_SLUGS.get(dataset, dataset)
    resolved_weight_mode = resolve_class_weight_mode(dataset, class_weight_mode, config)
    _, history, metrics = train_model(
        data=data,
        dataset_slug=dataset_slug,
        config=config,
=======
    data = prepare_dataset(dataset=dataset, data_dir=data_dir, label_column=label_column, max_samples=max_samples)
    dataset_slug = DATASET_SLUGS.get(dataset, dataset)
    resolved_weight_mode = resolve_class_weight_mode(dataset, class_weight_mode)
    _, history, metrics = train_model(
        data=data,
        dataset_slug=dataset_slug,
        epochs=epochs,
        batch_size=batch_size,
>>>>>>> e38505b (Update project with latest changes)
        results_dir=Path(results_dir),
        class_weight_mode=resolved_weight_mode,
        save_model=save_model,
    )

    return {
        "dataset_key": dataset,
        "dataset_slug": dataset_slug,
        "dataset_name": data.dataset_name,
        "label_column": data.label_column,
        "class_names": data.class_names,
        "class_distribution": data.class_distribution,
        "class_weight_mode": resolved_weight_mode,
<<<<<<< HEAD
        "training_config": config,
=======
>>>>>>> e38505b (Update project with latest changes)
        "history": history,
        "metrics": metrics,
    }


def write_outputs(
    experiments: Mapping[str, Mapping[str, object]],
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    combined: bool = False,
) -> None:
    """Write proposed-model metrics and figures without baseline artifacts."""

    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    for experiment in experiments.values():
        slug = str(experiment["dataset_slug"])
        single = {slug: experiment}
        save_metrics(single, results_path / f"{slug}_metrics.txt")
        plot_confusion_matrix(single, results_path / f"{slug}_confusion_matrix.png")
        plot_accuracy_curve(single, figures_path / f"{slug}_accuracy_curve.png")
        plot_loss_curve(single, figures_path / f"{slug}_loss_curve.png")

    if combined:
        save_metrics(experiments, results_path / "metrics.txt")
        plot_confusion_matrix(experiments, results_path / "confusion_matrix.png")
        plot_accuracy_curve(experiments, figures_path / "accuracy_curve.png")
        plot_loss_curve(experiments, figures_path / "loss_curve.png")
        plot_dataset_comparison(experiments, figures_path / "dataset_comparison_graph.png")


def run_project(
    dataset: str = "cic",
    all_datasets: bool = False,
    label_column: str | None = None,
    data_dir: str | Path = "dataset",
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
<<<<<<< HEAD
    epochs: int | None = None,
    batch_size: int | None = None,
=======
    epochs: int = 50,
    batch_size: int = 128,
>>>>>>> e38505b (Update project with latest changes)
    max_samples: int | None = None,
    class_weight_mode: str = "auto",
    save_model: bool = False,
    seed: int = 42,
) -> Dict[str, Mapping[str, object]]:
    """Run the requested proposed-model experiment and write artifacts."""

    dataset_keys = ("cic", "unsw") if all_datasets else (dataset,)
    experiments = {}
    for key in dataset_keys:
        experiments[key] = run_experiment(
            dataset=key,
            label_column=label_column if not all_datasets else None,
            data_dir=data_dir,
            results_dir=results_dir,
            epochs=epochs,
            batch_size=batch_size,
            max_samples=max_samples,
            class_weight_mode=class_weight_mode,
            save_model=save_model,
            seed=seed,
        )

    write_outputs(experiments, results_dir, figures_dir, combined=all_datasets)
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cic", "unsw"], default="cic")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--label-column", default=None)
<<<<<<< HEAD
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
=======
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
>>>>>>> e38505b (Update project with latest changes)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-weight-mode", choices=["auto", "none", "sqrt", "balanced"], default="auto")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_project(
        dataset=args.dataset,
        all_datasets=args.all_datasets,
        label_column=args.label_column,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        class_weight_mode=args.class_weight_mode,
        save_model=args.save_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()