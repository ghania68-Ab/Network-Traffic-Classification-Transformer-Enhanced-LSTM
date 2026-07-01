"""Data loading and preprocessing for network traffic classification.

The pipeline is designed to avoid data leakage: train/validation/test splits
are created before fitting imputers, encoders, scalers, or label encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.utils import resample

from config import TrainingConfig, get_training_config


RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed to load one supported dataset."""

    name: str
    label_column: str
    train_files: Tuple[str, ...]
    test_files: Tuple[str, ...] = ()
    drop_columns: Tuple[str, ...] = ()


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cic": DatasetSpec(
        name="CIC-Darknet2020",
        label_column="Label",
        train_files=("cicdarknet2020.parquet",),
        drop_columns=("Label.1",),
    ),
    "unsw": DatasetSpec(
        name="UNSW-NB15",
        label_column="attack_cat",
        train_files=("UNSW_NB15_training-set.parquet",),
        test_files=("UNSW_NB15_testing-set.parquet",),
        drop_columns=("label",),
    ),
}


@dataclass
class PreparedData:
    """Container returned by the preprocessing pipeline."""

    dataset_name: str
    label_column: str
    feature_names: List[str]
    class_names: List[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_distribution: Dict[str, Dict[str, int]]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.X_train.shape[1], self.X_train.shape[2]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def get_dataset_spec(dataset: str, label_column: Optional[str] = None) -> DatasetSpec:
    """Return a supported dataset spec, optionally overriding its label column."""

    key = dataset.lower()
    if key not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported values: {supported}.")

    spec = DATASET_SPECS[key]
    if label_column is None or label_column == spec.label_column:
        return spec

    drop_columns = tuple(c for c in spec.drop_columns if c != label_column)
    return DatasetSpec(
        name=spec.name,
        label_column=label_column,
        train_files=spec.train_files,
        test_files=spec.test_files,
        drop_columns=drop_columns,
    )


def load_dataset_frames(data_dir: Path, spec: DatasetSpec) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load parquet dataset files for a spec."""

    train_frames = [pd.read_parquet(data_dir / file_name) for file_name in spec.train_files]
    train_df = pd.concat(train_frames, ignore_index=True)

    test_df = None
    if spec.test_files:
        test_frames = [pd.read_parquet(data_dir / file_name) for file_name in spec.test_files]
        test_df = pd.concat(test_frames, ignore_index=True)

    return train_df, test_df


def clean_dataframe(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Remove duplicates and normalize missing, NaN, and infinite values."""

    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=[label_column])
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[label_column] != ""]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned

IDENTIFIER_COLUMNS = (
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Timestamp", "SimillarHTTP", "Unnamed: 0",
)


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-generalizable identifier columns that cause leakage or noise."""
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def cast_ports_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Treat port numbers as categories, not continuous magnitudes."""
    port_cols = [c for c in df.columns if c.lower() in {"src port", "dst port", "source port", "destination port"}]
    for col in port_cols:
        df[col] = df[col].astype(str)
    return df


def drop_train_test_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact test rows that also occur in train to avoid duplicate leakage."""

    train_hashes = set(pd.util.hash_pandas_object(train_df, index=False).astype("uint64"))
    test_hashes = pd.util.hash_pandas_object(test_df, index=False).astype("uint64")
    return test_df.loc[~test_hashes.isin(train_hashes)].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, label_column: str, max_samples: Optional[int]) -> pd.DataFrame:
    """Return a reproducible stratified sample for fast development runs."""

    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df

    sampled_parts = []
    label_counts = df[label_column].value_counts()
    for label, count in label_counts.items():
        target = max(1, int(round(max_samples * count / len(df))))
        if count >= 2:
            target = max(2, target)
        target = min(target, count)
        sampled_parts.append(
            resample(
                df[df[label_column] == label],
                replace=False,
                n_samples=target,
                random_state=RANDOM_STATE,
            )
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)
    # Keep rare-class examples even if the rounded allocation slightly exceeds
    # max_samples; dropping them can break stratified validation/test splitting.
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def safe_stratify(df: pd.DataFrame, label_column: str):
    """Return labels for stratification only when every class has enough rows."""

    counts = df[label_column].value_counts()
    return df[label_column] if len(counts) > 1 and counts.min() >= 2 else None


def split_frames(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    label_column: str,
    validation_size: float,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test dataframes without leaking test data."""

    if test_df is not None:
        train_part, val_part = train_test_split(
            train_df,
            test_size=validation_size,
            random_state=RANDOM_STATE,
            stratify=safe_stratify(train_df, label_column),
        )
        return train_part.reset_index(drop=True), val_part.reset_index(drop=True), test_df.reset_index(drop=True)

    train_part, temp_part = train_test_split(
        train_df,
        test_size=validation_size + test_size,
        random_state=RANDOM_STATE,
        stratify=safe_stratify(train_df, label_column),
    )
    relative_test_size = test_size / (validation_size + test_size)
    val_part, test_part = train_test_split(
        temp_part,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=safe_stratify(temp_part, label_column),
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True), test_part.reset_index(drop=True)


def find_correlated_columns(df: pd.DataFrame, threshold: float = 0.97) -> List[str]:
    """Return numeric columns that are highly correlated with an earlier column."""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) < 2:
        return []

    correlation = df[numeric_columns].corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    return [column for column in upper.columns if upper[column].gt(threshold).any()]


def apply_feature_filters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    correlation_threshold: float,
    variance_threshold: float,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove noisy numeric columns using train-only statistics."""

    train_features = train_df[feature_columns].copy()
    correlated = find_correlated_columns(train_features, threshold=correlation_threshold)
    remaining = [column for column in feature_columns if column not in correlated]

    numeric_remaining = train_features[remaining].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_remaining:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(train_features[numeric_remaining])
        kept_numeric = [
            column
            for column, keep in zip(numeric_remaining, selector.get_support())
            if keep
        ]
        categorical_remaining = [column for column in remaining if column not in numeric_remaining]
        remaining = categorical_remaining + kept_numeric

    if not remaining:
        raise ValueError("Feature filtering removed every input column.")

    return (
        remaining,
        train_df[remaining + [label_column]].copy(),
        val_df[remaining + [label_column]].copy(),
        test_df[remaining + [label_column]].copy(),
    )


def oversample_training_frame(
    train_df: pd.DataFrame,
    label_column: str,
    target_ratio: float,
) -> pd.DataFrame:
    """Oversample minority classes on the training split only."""

    if target_ratio <= 0:
        return train_df

    label_counts = train_df[label_column].value_counts()
    majority_count = int(label_counts.max())
    target_count = max(2, int(round(majority_count * target_ratio)))

    parts = []
    for label, count in label_counts.items():
        class_frame = train_df[train_df[label_column] == label]
        if count < target_count:
            extra = resample(
                class_frame,
                replace=True,
                n_samples=target_count - count,
                random_state=RANDOM_STATE,
            )
            parts.append(pd.concat([class_frame, extra], ignore_index=True))
        else:
            parts.append(class_frame)

    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a fit-on-train-only preprocessing transformer."""

    categorical_columns = [
        column
        for column in X_train.columns
        if str(X_train[column].dtype) in {"object", "category", "bool"}
    ]
    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def prepare_lstm_features(array: np.ndarray, group_size: int = 8) -> np.ndarray:
    """Convert tabular features to grouped LSTM input shape.

    Grouping related flow features into short multivariate timesteps gives the
    LSTM richer local context than treating every feature as a scalar timestep.
    """

    array = np.asarray(array, dtype=np.float32)
    n_samples, n_features = array.shape
    group_size = max(2, min(group_size, n_features))

    pad_len = (group_size - (n_features % group_size)) % group_size
    if pad_len:
        array = np.pad(array, ((0, 0), (0, pad_len)), mode="constant")

    n_timesteps = array.shape[1] // group_size
    return array.reshape(n_samples, n_timesteps, group_size)


def class_distribution(*splits: Tuple[str, Iterable[int]], class_names: List[str]) -> Dict[str, Dict[str, int]]:
    """Return readable class counts for each split."""

    distribution: Dict[str, Dict[str, int]] = {}
    for split_name, labels in splits:
        counts = pd.Series(labels).value_counts().sort_index()
        distribution[split_name] = {
            class_names[int(index)]: int(value) for index, value in counts.items()
        }
    return distribution


def prepare_dataset(
    dataset: str = "cic",
    data_dir: str | Path = "dataset",
    label_column: Optional[str] = None,
    validation_size: float = 0.10,
    test_size: float = 0.10,
    max_samples: Optional[int] = None,
    training_config: TrainingConfig | None = None,
    oversample: bool | None = None,
) -> PreparedData:
    """Load, clean, split, encode, scale, and reshape a supported dataset."""

    config = training_config or get_training_config(dataset)
    apply_oversample = config.oversample_ratio > 0 if oversample is None else oversample

    data_path = Path(data_dir)
    spec = get_dataset_spec(dataset, label_column)
    train_df, test_df = load_dataset_frames(data_path, spec)

    if spec.label_column not in train_df.columns:
        raise ValueError(f"Label column '{spec.label_column}' was not found in {spec.name}.")

    train_df = clean_dataframe(train_df, spec.label_column)
    train_df = drop_identifier_columns(train_df)
    train_df = cast_ports_to_categorical(train_df)
    if test_df is not None:
        test_df = clean_dataframe(test_df, spec.label_column)
        test_df = drop_identifier_columns(test_df)
        test_df = cast_ports_to_categorical(test_df)
        test_df = drop_train_test_overlap(train_df, test_df)
    else:
        train_df = stratified_sample(train_df, spec.label_column, max_samples)

    if test_df is not None and max_samples is not None and max_samples > 0:
        train_target = max(2, int(round(max_samples * 0.70)))
        test_target = max(2, max_samples - train_target)
        train_df = stratified_sample(train_df, spec.label_column, train_target)
        test_df = stratified_sample(test_df, spec.label_column, test_target)

    train_split, val_split, test_split = split_frames(
        train_df=train_df,
        test_df=test_df,
        label_column=spec.label_column,
        validation_size=validation_size,
        test_size=test_size,
    )

    feature_drop_columns = {spec.label_column, *spec.drop_columns}
    feature_columns = [column for column in train_split.columns if column not in feature_drop_columns]

    feature_columns, train_split, val_split, test_split = apply_feature_filters(
        train_df=train_split,
        val_df=val_split,
        test_df=test_split,
        feature_columns=feature_columns,
        label_column=spec.label_column,
        correlation_threshold=config.correlation_threshold,
        variance_threshold=config.variance_threshold,
    )

    if apply_oversample:
        train_split = oversample_training_frame(
            train_split,
            label_column=spec.label_column,
            target_ratio=config.oversample_ratio,
        )

    X_train_df = train_split[feature_columns]
    X_val_df = val_split[feature_columns]
    X_test_df = test_split[feature_columns]

    y_train_raw = train_split[spec.label_column].astype(str)
    y_val_raw = val_split[spec.label_column].astype(str)
    y_test_raw = test_split[spec.label_column].astype(str)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    preprocessor = build_preprocessor(X_train_df)
    X_train = prepare_lstm_features(
        preprocessor.fit_transform(X_train_df),
        group_size=config.feature_group_size,
    )
    X_val = prepare_lstm_features(
        preprocessor.transform(X_val_df),
        group_size=config.feature_group_size,
    )
    X_test = prepare_lstm_features(
        preprocessor.transform(X_test_df),
        group_size=config.feature_group_size,
    )

    class_names = [str(label) for label in label_encoder.classes_]
    distribution = class_distribution(
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
        class_names=class_names,
    )

    try:
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        feature_names = [str(column) for column in feature_columns]

    return PreparedData(
        dataset_name=spec.name,
        label_column=spec.label_column,
        feature_names=feature_names,
        class_names=class_names,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        class_distribution=distribution,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
    )

