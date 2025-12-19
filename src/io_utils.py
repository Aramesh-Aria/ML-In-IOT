from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle

from . import config

def ensure_dirs() -> None:
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(prefer_processed: bool = True) -> pd.DataFrame:
    path = config.DATA_PROCESSED if (prefer_processed and config.DATA_PROCESSED.exists()) else config.DATA_RAW
    df = pd.read_csv(path)

    # Drop accidental index column(s)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    if "" in df.columns:
        df = df.drop(columns=[""])
    return df

def detect_target_col(df: pd.DataFrame) -> str:
    if config.TARGET_COL and config.TARGET_COL in df.columns:
        return config.TARGET_COL

    # fallback by candidates
    for c in config.TARGET_CANDIDATES:
        if c in df.columns:
            return c

    # last resort: if exactly one non-feature-like column? (avoid guessing too hard)
    raise ValueError(
        "Target column not found. Please set TARGET_COL in src/config.py. "
        f"Columns available: {df.columns.tolist()}"
    )

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    target = detect_target_col(df)

    cols_to_drop = [c for c in getattr(config, "DROP_COLS", []) if c in df.columns]
    df2 = df.drop(columns=cols_to_drop)

    X = df2.drop(columns=[target])
    y = df2[target]
    return X, y, target

def load_model(model_path: Path):
    # try joblib first
    try:
        return joblib.load(model_path)
    except Exception:
        # fallback pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)

def align_features_for_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align X columns to what the model expects.
    If model has feature_names_in_, use it.
    Otherwise, use X as-is.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        if missing:
            raise ValueError(f"Missing columns required by model: {missing}")
        # Reorder and drop extra safely
        X_aligned = X[expected].copy()
        return X_aligned

    return X

def safe_numeric_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure X is numeric (convert where possible).
    """
    X2 = X.copy()
    for col in X2.columns:
        if not np.issubdtype(X2[col].dtype, np.number):
            X2[col] = pd.to_numeric(X2[col], errors="coerce")
    # Fill NaNs with median (simple, fast)
    X2 = X2.fillna(X2.median(numeric_only=True))
    return X2

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
