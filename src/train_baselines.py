# src/train_baselines.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from . import config
from .io_utils import ensure_dirs, load_dataset, split_xy, safe_numeric_X, save_csv

TRAINED_MODELS_DIR = Path("models_trained")

DROP_COLS_DEFAULT = ["num", "timestamp", "device_id", "counter"]  # قابل تغییر

def main():
    ensure_dirs()
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(prefer_processed=False)  # فعلاً raw
    # حذف ستون‌های غیرلازم اگر وجود دارند
    drop_cols = [c for c in DROP_COLS_DEFAULT if c in df.columns]
    df = df.drop(columns=drop_cols)

    X, y, target = split_xy(df)
    X = safe_numeric_X(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    models = {
        "ridge": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "rf": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, gamma="scale", epsilon=0.1)),
        ]),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        rows.append({"model": name, "rmse": rmse, "r2": r2})

        joblib.dump(model, TRAINED_MODELS_DIR / f"{name}.joblib")

    metrics = pd.DataFrame(rows).sort_values("rmse")
    save_csv(metrics, config.MODEL_METRICS_CSV)
    print(metrics)
    print("Saved models to:", TRAINED_MODELS_DIR)

if __name__ == "__main__":
    main()
