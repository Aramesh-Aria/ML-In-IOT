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
    "ridge.joblib",
    "svr.joblib",
    "rf.joblib",
]



# مدل منتخب
SELECTED_TRAINED_MODEL = "ridge.joblib"  # چون بهترین rmse را دارد

# مسیر مدل‌های آموزش‌داده‌شده
TRAINED_MODELS_DIR = PROJECT_ROOT / "models_trained"

# محدوده‌های LoRa (برای TPC)
SF_MIN, SF_MAX = 7, 12
TP_MIN, TP_MAX = 2, 14  # dBm (برای اروپا/LoRa رایج است؛ اگر منطقه‌تان متفاوت است، اصلاح کنید)

# Link Margin (فرض عملی برای ارائه؛ اگر مقاله مقدار مشخص دارد، همان را جایگزین کنید)
LINK_MARGIN_DB = 10.0

# جدول SNR_limit بر اساس SF (مقادیر رایج برای BW=125kHz؛ برای ارائه کفایت دارد)
SNR_LIMIT_BY_SF = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}

# baseline ثابت برای مقایسه
BASELINE_SF = 12
BASELINE_TP = 14
LORA_BW_HZ = 125_000
