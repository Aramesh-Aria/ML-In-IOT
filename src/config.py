from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "subsampled_data.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "dataset_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_METRICS_CSV = PRED_DIR / "model_metrics.csv"
SNR_PREDICTIONS_CSV = PRED_DIR / "snr_predictions.csv"
TPC_DECISIONS_CSV = PRED_DIR / "tpc_decisions.csv"

# --- Experiment settings ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Target detection (fallback) ---
# اگر نام ستون هدف را دقیق می‌دانید، TARGET_COL را مقداردهی کنید (مثلاً "SNR")
TARGET_COL = "snr"
TARGET_CANDIDATES = ["snr"]
#ستون هایی که میخوایم حذف کنیم
DROP_COLS = ["num","timestamp", "device_id", "counter"]

# --- Models to focus on (2–3 models only) ---
PRIMARY_MODELS = [
    "model_random_forest.sav",
    "model_svr.sav",
    "model_ridge.sav",
]

# اگر بخواهید به‌جای این‌ها، MLR را هم داشته باشید:
# PRIMARY_MODELS = ["model_random_forest.sav", "model_svr.sav", "model_mlr.sav"]
