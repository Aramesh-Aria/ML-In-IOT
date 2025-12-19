from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from . import config
from .io_utils import ensure_dirs, load_dataset, split_xy, load_model, align_features_for_model, safe_numeric_X, save_csv

def evaluate_one(model_file: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model_path = config.MODELS_DIR / model_file
    model = load_model(model_path)

    X_aligned = align_features_for_model(X_test, model)
    X_aligned = safe_numeric_X(X_aligned)

    y_pred = model.predict(X_aligned)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "model_file": model_file,
        "rmse": rmse,
        "r2": r2,
    }

def main():
    ensure_dirs()

    df = load_dataset(prefer_processed=True)
    X, y, target = split_xy(df)

    # Split only once so all models are compared fairly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    rows = []
    for mf in config.PRIMARY_MODELS:
        try:
            rows.append(evaluate_one(mf, X_test, y_test))
        except Exception as e:
            rows.append({
                "model_file": mf,
                "rmse": None,
                "r2": None,
                "error": str(e),
            })

    out = pd.DataFrame(rows).sort_values(by="rmse", na_position="last")
    save_csv(out, config.MODEL_METRICS_CSV)

    print("Saved:", config.MODEL_METRICS_CSV)
    print(out)

if __name__ == "__main__":
    main()
