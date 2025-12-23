"""

هدف این فایل:
- متمرکز کردن کارهای تکراری و عمومی مربوط به:
  1) ساخت پوشه‌های خروجی
  2) بارگذاری دیتاست (raw یا processed)
  3) تشخیص ستون هدف (Target)
  4) جداسازی X و y
  5) بارگذاری مدل (joblib یا pickle)
  6) همسان‌سازی ستون‌های ورودی با مدل (feature alignment)
  7) تبدیل امن ویژگی‌ها به عددی (numeric coercion)
  8) ذخیره فایل‌های CSV

مزیت:
- کدهای اصلی مثل run_pipeline.py و train_baselines.py تمیز می‌مانند
- هر تغییری در روش load/save یا پیش‌پردازش ورودی‌ها فقط در یک فایل انجام می‌شود
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle

from . import config


def ensure_dirs() -> None:
    """
    ساخت پوشه‌های خروجی استاندارد پروژه.

    چرا لازم است؟
    - بسیاری از اسکریپت‌ها (run_pipeline, evaluate_models, ...) خروجی تولید می‌کنند.
    - اگر پوشه‌ها وجود نداشته باشند، ذخیره CSV/PNG خطا می‌دهد.
    - این تابع تضمین می‌کند مسیرهای خروجی همیشه آماده‌اند.
    """
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(prefer_processed: bool = True) -> pd.DataFrame:
    """
    بارگذاری دیتاست از data/raw یا data/processed.

    منطق انتخاب فایل:
    - اگر prefer_processed=True و فایل processed وجود داشته باشد => processed را می‌خوانیم
    - در غیر این صورت => raw را می‌خوانیم

    پاک‌سازی‌های کوچک هنگام خواندن:
    - حذف ستون‌های ناخواسته‌ای مثل "Unnamed: 0" که معمولاً به عنوان ستون ایندکس ذخیره می‌شوند.
    - حذف ستون خالی با نام "" در صورت وجود (گاهی به دلیل ذخیره بد CSV ایجاد می‌شود)

    Fallback مهم:
    - اگر فایل processed وجود داشته باشد ولی ستون هدف (مثلاً snr) داخلش نباشد،
      آن را «خراب/ناقص» فرض می‌کنیم و به raw برمی‌گردیم.
    """
    # انتخاب مسیر دیتاست بر اساس prefer_processed و وجود فایل processed
    path = config.DATA_PROCESSED if (prefer_processed and config.DATA_PROCESSED.exists()) else config.DATA_RAW
    df = pd.read_csv(path)

    # حذف ستون‌های ایندکس تصادفی مانند Unnamed: 0
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

    # حذف ستون با نام خالی "" اگر موجود باشد
    if "" in df.columns:
        df = df.drop(columns=[""])

    # --- Fallback: اگر processed ستون هدف را نداشت، از raw استفاده می‌کنیم ---
    target = config.TARGET_COL or None
    if target and target not in df.columns:
        df = pd.read_csv(config.DATA_RAW)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
        if "" in df.columns:
            df = df.drop(columns=[""])

    return df


def detect_target_col(df: pd.DataFrame) -> str:
    """
    تشخیص ستون هدف (Target) برای مدل ML.

    اولویت‌ها:
    1) اگر config.TARGET_COL ست شده باشد و در ستون‌های df موجود باشد => همان را برمی‌گرداند.
    2) در غیر این صورت، از لیست config.TARGET_CANDIDATES به ترتیب استفاده می‌کند.
    3) اگر هیچ‌کدام پیدا نشد => خطا می‌دهد (به جای حدس زدن بی‌پایه)

    دلیل خطا دادن:
    - حدس زدن Target می‌تواند باعث Training اشتباه و نتایج گمراه‌کننده شود.
    """
    # حالت اصلی: کاربر دقیقاً TARGET_COL را مشخص کرده است
    if config.TARGET_COL and config.TARGET_COL in df.columns:
        return config.TARGET_COL

    # حالت fallback: جستجو در کاندیدها
    for c in config.TARGET_CANDIDATES:
        if c in df.columns:
            return c

    # اگر target پیدا نشود، به کاربر اطلاع می‌دهیم تا config را اصلاح کند
    raise ValueError(
        "Target column not found. Please set TARGET_COL in src/config.py. "
        f"Columns available: {df.columns.tolist()}"
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    جداسازی ویژگی‌ها (X) و هدف (y) از دیتافریم.

    مراحل:
    1) تشخیص ستون هدف با detect_target_col
    2) حذف ستون‌هایی که در config.DROP_COLS مشخص شده‌اند (اگر وجود داشته باشند)
       این ستون‌ها معمولاً شناسه‌ای/زمانی/متنی هستند و بدون مهندسی ویژگی مناسب نیستند.
    3) ساخت X با حذف ستون target
    4) ساخت y برابر ستون target

    خروجی:
    - X: DataFrame ویژگی‌ها
    - y: Series هدف
    - target: نام ستون هدف (برای گزارش/دیباگ)
    """
    target = detect_target_col(df)

    # حذف ستون‌های DROP_COLS در صورت وجود (به‌صورت امن)
    cols_to_drop = [c for c in getattr(config, "DROP_COLS", []) if c in df.columns]
    df2 = df.drop(columns=cols_to_drop)

    # X: تمام ستون‌ها به جز target
    X = df2.drop(columns=[target])
    # y: فقط ستون target
    y = df2[target]

    return X, y, target


def load_model(model_path: Path):
    """
    بارگذاری مدل از روی دیسک.

    دلیل این طراحی:
    - برخی مدل‌ها ممکن است با joblib ذخیره شده باشند
    - برخی دیگر ممکن است pickle باشند (یا مدل‌های قدیمی مقاله)
    - ابتدا joblib تلاش می‌کنیم چون در پروژه‌های sklearn رایج‌تر و سریع‌تر است.
    - اگر شکست خورد، به pickle fallback می‌کنیم.

    خروجی:
    - مدل لود شده (ابجکت sklearn یا مشابه)
    """
    # try joblib first
    try:
        return joblib.load(model_path)
    except Exception:
        # fallback pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)


def align_features_for_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    همسان‌سازی ستون‌های X با ستون‌هایی که مدل انتظار دارد.

    مشکل رایج:
    - برخی مدل‌های sklearn هنگام آموزش، نام ستون‌ها را ذخیره می‌کنند (feature_names_in_).
    - اگر در زمان predict ستون‌ها ترتیب متفاوت داشته باشند یا ستون اضافی/کمبود داشته باشیم،
      ممکن است خطا یا نتایج نادرست بگیریم.

    راه‌حل:
    - اگر مدل attribute به نام feature_names_in_ داشته باشد:
        * ستون‌های مورد انتظار را استخراج می‌کنیم
        * اگر ستونی کم باشد => خطا می‌دهیم (چون مدل نمی‌تواند درست کار کند)
        * ستون‌های اضافه را حذف می‌کنیم
        * ستون‌ها را دقیقاً به ترتیب مورد انتظار مدل مرتب می‌کنیم
    - اگر مدل feature_names_in_ نداشت:
        * ورودی X را همان‌طور برمی‌گردانیم (فرض می‌کنیم خود کاربر/پایپ‌لاین درست است)
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]

        if missing:
            raise ValueError(f"Missing columns required by model: {missing}")

        # ستون‌های اضافی را حذف و ترتیب را مطابق expected تنظیم می‌کنیم
        X_aligned = X[expected].copy()
        return X_aligned

    return X


def safe_numeric_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    تبدیل امن ویژگی‌ها به عددی (numeric) برای جلوگیری از خطا در sklearn.

    چرا لازم است؟
    - اگر ستون‌هایی مثل device_id یا timestamp وارد مدل شوند (یا به اشتباه باقی بمانند)
      sklearn نمی‌تواند روی داده‌های غیرعددی پیش‌بینی کند.
    - حتی برخی ستون‌های عددی ممکن است به صورت رشته (string) خوانده شوند.

    رفتار تابع:
    1) برای هر ستون غیرعددی تلاش می‌کند آن را به عدد تبدیل کند (pd.to_numeric)
       اگر تبدیل نشد => NaN می‌شود
    2) در پایان، NaNها را با میانه ستون پر می‌کند (روش سریع/ساده)
       (این روش برای ارائه و dataset کوچک مناسب است)
    """
    X2 = X.copy()

    # تلاش برای تبدیل ستون‌های غیرعددی به عددی
    for col in X2.columns:
        if not np.issubdtype(X2[col].dtype, np.number):
            X2[col] = pd.to_numeric(X2[col], errors="coerce")

    # پر کردن NaNها با میانه ستون‌های عددی
    X2 = X2.fillna(X2.median(numeric_only=True))
    return X2


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    ذخیره یک DataFrame در مسیر مشخص شده به صورت CSV.

    ویژگی:
    - مسیر پوشه را اگر وجود نداشته باشد می‌سازد
    - index=False برای جلوگیری از ذخیره ایندکس اضافی در CSV
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
