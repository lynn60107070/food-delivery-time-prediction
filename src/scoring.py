"""Load serialized sklearn pipelines and score raw delivery rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import preprocess_for_scoring
from src.flaml_wrapper import register_notebook_pickles
from src.model_config import DROP_COLS, TARGET


def load_metadata(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: Path | str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p.resolve()}")
    register_notebook_pickles()
    return joblib.load(p)


def build_feature_matrix(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    preprocessed: bool = False,
) -> pd.DataFrame:
    """Raw rows -> feature matrix expected by the saved sklearn Pipeline.

    If ``preprocessed=True``, ``df`` is already run through ``preprocess_for_scoring`` (e.g. after
    simulation tweaks); skip preprocessing and only drop columns / select features.
    """
    out = df if preprocessed else preprocess_for_scoring(df)
    drop_cols = [c for c in DROP_COLS if c in out.columns]
    out = out.drop(columns=drop_cols)
    if TARGET in out.columns:
        out = out.drop(columns=[TARGET])
    feature_cols = None
    if metadata and "selected_features_used" in metadata:
        feature_cols = list(metadata["selected_features_used"])
    elif metadata and "feature_columns" in metadata:
        feature_cols = list(metadata["feature_columns"])
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")
        out = out[feature_cols].copy()
    return out


def align_to_pipeline(X: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    if hasattr(pipeline, "feature_names_in_"):
        expected = list(pipeline.feature_names_in_)
        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns for model: {sorted(missing)}")
        return X[expected]
    return X


def predict_delivery_time(
    df: pd.DataFrame,
    pipeline: Any,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    X = build_feature_matrix(df, metadata=metadata)
    X = align_to_pipeline(X, pipeline)
    return np.asarray(pipeline.predict(X), dtype=float)


def predict_delivery_time_preprocessed(
    preprocessed_df: pd.DataFrame,
    pipeline: Any,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predict from an already preprocessed feature frame (same columns as after ``preprocess_for_scoring``)."""
    X = build_feature_matrix(preprocessed_df, metadata=metadata, preprocessed=True)
    X = align_to_pipeline(X, pipeline)
    return np.asarray(pipeline.predict(X), dtype=float)


def add_predicted_sla_status(
    df: pd.DataFrame,
    pred_col: str = "predicted_delivery_time_min",
    sla_minutes: float = 30.0,
) -> pd.DataFrame:
    """Label each row On-Time vs Delayed from predicted minutes vs SLA threshold."""
    out = df.copy()
    out["predicted_SLA_status"] = out[pred_col].apply(
        lambda x: "On-Time" if x <= sla_minutes else "Delayed"
    )
    return out
