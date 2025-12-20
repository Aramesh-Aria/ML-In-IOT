import pandas as pd
from src import config

def top_counts(series: pd.Series, n: int = 5):
    vc = series.value_counts().head(n)
    return {str(k): int(v) for k, v in vc.items()}

def main():
    dec = pd.read_csv(config.TPC_DECISIONS_CSV)

    # Basic sanity
    required = {"sf_new", "tp_new", "me", "energy_norm"}
    missing = required - set(dec.columns)
    if missing:
        raise ValueError(f"Missing required columns in tpc_decisions.csv: {missing}")

    summary = {
        "count": int(len(dec)),

        # Energy
        "energy_norm_mean": float(dec["energy_norm"].mean()),
        "energy_norm_median": float(dec["energy_norm"].median()),
        "pct_energy_below_1": float((dec["energy_norm"] < 1.0).mean() * 100),
        "pct_energy_below_0_5": float((dec["energy_norm"] < 0.5).mean() * 100),

        # SF
        "sf_mode": int(dec["sf_new"].mode().iloc[0]),
        "sf_min": int(dec["sf_new"].min()),
        "sf_max": int(dec["sf_new"].max()),
        "sf_top_counts": top_counts(dec["sf_new"], n=6),

        # TP
        "tp_mean": float(dec["tp_new"].mean()),
        "tp_median": float(dec["tp_new"].median()),
        "tp_min": float(dec["tp_new"].min()),
        "tp_max": float(dec["tp_new"].max()),
        "tp_top_counts": top_counts(dec["tp_new"].round(1), n=8),

        # Margin
        "me_mean": float(dec["me"].mean()),
        "me_median": float(dec["me"].median()),
        "pct_me_ge_0": float((dec["me"] >= 0.0).mean() * 100),
    }

    # Optional: round for nicer printing
    pretty = {}
    for k, v in summary.items():
        if isinstance(v, float):
            pretty[k] = round(v, 4)
        else:
            pretty[k] = v

    print(pretty)

if __name__ == "__main__":
    main()
