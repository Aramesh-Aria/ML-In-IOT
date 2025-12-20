from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from . import config
from .io_utils import ensure_dirs, load_dataset, detect_target_col, safe_numeric_X, save_csv
from .tpc import decide_tpc
from .energy import normalized_energy

def main():
    ensure_dirs()

    df = load_dataset(prefer_processed=False).copy()

    # Drop non-ML columns
    drop_cols = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    target = detect_target_col(df)
    X = df.drop(columns=[target]).copy()
    y_true = df[target].copy()

    # Load trained model
    model_path = config.TRAINED_MODELS_DIR / config.SELECTED_TRAINED_MODEL
    model = joblib.load(model_path)

    Xn = safe_numeric_X(X)
    snr_pred = model.predict(Xn)

    # Save predictions
    pred_df = pd.DataFrame({
        "snr_true": y_true.values,
        "snr_pred": snr_pred,
    })
    save_csv(pred_df, config.SNR_PREDICTIONS_CSV)

    # TPC decisions per row
    decisions = []
    for sp in snr_pred:
        d = decide_tpc(float(sp))
        decisions.append({
            "sf_new": d.sf,
            "tp_new": d.tp,
            "me": d.me,
            "energy_norm": normalized_energy(d.tp, d.sf, tp_ref=config.BASELINE_TP, sf_ref=config.BASELINE_SF),
        })
    dec_df = pd.DataFrame(decisions)
    save_csv(dec_df, config.TPC_DECISIONS_CSV)

    # --- Figures for report ---
    # 1) True vs Pred
    plt.figure()
    plt.scatter(pred_df["snr_true"], pred_df["snr_pred"])
    plt.xlabel("SNR true (dB)")
    plt.ylabel("SNR predicted (dB)")
    plt.title("SNR: True vs Predicted")
    plt.savefig(config.FIG_DIR / "snr_true_vs_pred.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Distribution of chosen SF
    plt.figure()
    dec_df["sf_new"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("SF chosen")
    plt.ylabel("Count")
    plt.title("TPC Output: SF Distribution")
    plt.savefig(config.FIG_DIR / "sf_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Distribution of chosen TP
    plt.figure()
    plt.hist(dec_df["tp_new"], bins=13)  # مناسب برای بازه 2..14 با گام 1
    plt.xlabel("TP chosen (dBm)")
    plt.ylabel("Count")
    plt.title("TPC Output: TP Distribution")
    plt.savefig(config.FIG_DIR / "tp_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Margin (Me) histogram
    plt.figure()
    plt.hist(dec_df["me"], bins=20)
    plt.xlabel("Margin Me (dB)")
    plt.ylabel("Count")
    plt.title("Margin Distribution after TPC")
    plt.savefig(config.FIG_DIR / "me_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 5) Energy norm histogram
    plt.figure()
    plt.hist(dec_df["energy_norm"], bins=20)
    plt.xlabel("Normalized energy (vs baseline SF=12, TP=14)")
    plt.ylabel("Count")
    plt.title("Energy Reduction Proxy")
    plt.savefig(config.FIG_DIR / "energy_norm_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved predictions:", config.SNR_PREDICTIONS_CSV)
    print("Saved decisions:", config.TPC_DECISIONS_CSV)
    print("Saved figures in:", config.FIG_DIR)

if __name__ == "__main__":
    main()
