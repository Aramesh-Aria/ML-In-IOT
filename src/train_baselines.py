"""
هدف این اسکریپت:
- آموزش مجدد چند مدل baseline (Ridge / RandomForest / SVR) روی دیتاست پروژه
- ارزیابی منصفانه آن‌ها روی یک Test split ثابت (با random_state مشخص)
- ذخیره مدل‌های آموزش‌داده‌شده در پوشه models_trained/ به صورت joblib
- ذخیره جدول متریک‌ها (RMSE و R²) در outputs/predictions/model_metrics.csv

چرا این فایل مهم است؟
- مدل‌های آماده (.sav) مقاله در محیط شما با نسخه‌های جدید sklearn مشکل داشتند.
- بنابراین مدل‌ها را در محیط فعلی بازآموزی کردید تا:
  1) قابل اجرا باشند
  2) نتایج قابل تکرار داشته باشند
  3) بتوانید مدل منتخب را با معیار علمی انتخاب کنید
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ابزارهای استاندارد آموزش/ارزیابی
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# برای SVR به scaling نیاز داریم (چون حساس به مقیاس ویژگی‌هاست)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# مدل‌ها
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from . import config
from .io_utils import ensure_dirs, load_dataset, split_xy, safe_numeric_X, save_csv


# -----------------------------------------------------------------------------
# محل ذخیره مدل‌های آموزش‌داده‌شده توسط این اسکریپت
# نکته: بهتر است از config.TRAINED_MODELS_DIR استفاده شود تا یک‌دست باشد،
# ولی فعلاً همین هم درست کار می‌کند چون مسیر پروژه شما همین است.
# -----------------------------------------------------------------------------
TRAINED_MODELS_DIR = Path("models_trained")


# -----------------------------------------------------------------------------
# ستون‌های پیش‌فرضی که برای ML مناسب نیستند (شناسه/زمان/متن)
# در پروژه شما همین‌ها در config.DROP_COLS هم تعریف شده‌اند.
# -----------------------------------------------------------------------------
DROP_COLS_DEFAULT = ["num", "timestamp", "device_id", "counter"]  # قابل تغییر


def main():
    """
    اجرای کامل آموزش و ارزیابی baselineها.

    مراحل:
    1) آماده‌سازی پوشه‌های خروجی (ensure_dirs)
    2) بارگذاری دیتاست خام
    3) حذف ستون‌های غیرمفید برای ML
    4) جداسازی X و y (هدف: snr)
    5) تبدیل ویژگی‌ها به عددی (safe_numeric_X)
    6) train/test split ثابت
    7) تعریف مدل‌ها و آموزش هر کدام
    8) ارزیابی روی Test set با RMSE و R²
    9) ذخیره مدل‌ها در models_trained/
    10) ذخیره جدول متریک‌ها در outputs/predictions/model_metrics.csv
    """
    # -------------------------------------------------------------------------
    # 1) ساخت پوشه‌های خروجی (outputs/...) و پوشه مدل‌های آموزش‌داده‌شده
    # -------------------------------------------------------------------------
    ensure_dirs()
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) بارگذاری دیتاست
    # prefer_processed=False یعنی از raw استفاده می‌کنیم (در پروژه شما processed فعلاً استفاده نمی‌شود)
    # -------------------------------------------------------------------------
    df = load_dataset(prefer_processed=False)

    # -------------------------------------------------------------------------
    # 3) حذف ستون‌های غیرلازم در صورت وجود
    # (این کار ریسک ورود ستون‌های غیرعددی یا شناسه‌ای به مدل را کم می‌کند)
    # -------------------------------------------------------------------------
    drop_cols = [c for c in DROP_COLS_DEFAULT if c in df.columns]
    df = df.drop(columns=drop_cols)

    # -------------------------------------------------------------------------
    # 4) split_xy:
    # X = ویژگی‌ها
    # y = ستون هدف (snr)
    # target = نام ستون هدف برای اطلاع
    # -------------------------------------------------------------------------
    X, y, target = split_xy(df)

    # -------------------------------------------------------------------------
    # 5) تبدیل X به عددی و مدیریت NaN
    # این مرحله مطمئن می‌کند sklearn با خطای dtype مواجه نمی‌شود.
    # -------------------------------------------------------------------------
    X = safe_numeric_X(X)

    # -------------------------------------------------------------------------
    # 6) تقسیم Train/Test ثابت برای مقایسه منصفانه
    # تمام مدل‌ها دقیقاً روی یک Test set ارزیابی می‌شوند
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    # -------------------------------------------------------------------------
    # 7) تعریف مدل‌ها
    #
    # Ridge:
    # - یک baseline خطی و سریع
    # - مناسب برای داده‌های کوچک، پایدار و قابل توضیح
    #
    # RandomForest:
    # - مدل غیرخطی قوی برای روابط پیچیده
    # - معمولاً نیاز به scaling ندارد
    #
    # SVR:
    # - به scaling حساس است، پس با StandardScaler در Pipeline قرار داده شده
    # -------------------------------------------------------------------------
    models = {
        "ridge": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, gamma="scale", epsilon=0.1)),
        ]),
    }

    rows = []

    # -------------------------------------------------------------------------
    # 8) حلقه آموزش + ارزیابی + ذخیره مدل
    # -------------------------------------------------------------------------
    for name, model in models.items():
        # آموزش مدل
        model.fit(X_train, y_train)

        # پیش‌بینی روی Test
        pred = model.predict(X_test)

        # محاسبه متریک‌ها
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))

        # ذخیره متریک برای جدول خروجی
        rows.append({"model": name, "rmse": rmse, "r2": r2})

        # ذخیره مدل آموزش‌داده‌شده برای استفاده در run_pipeline.py
        joblib.dump(model, TRAINED_MODELS_DIR / f"{name}.joblib")

    # -------------------------------------------------------------------------
    # 9) ساخت جدول نتایج و مرتب‌سازی بر اساس RMSE (کمتر بهتر)
    # -------------------------------------------------------------------------
    metrics = pd.DataFrame(rows).sort_values("rmse")

    # -------------------------------------------------------------------------
    # 10) ذخیره متریک‌ها در خروجی استاندارد پروژه
    # -------------------------------------------------------------------------
    save_csv(metrics, config.MODEL_METRICS_CSV)

    # چاپ نتایج برای مشاهده سریع در ترمینال/نوت‌بوک
    print(metrics)
    print("Saved models to:", TRAINED_MODELS_DIR)


if __name__ == "__main__":
    # اجرای مستقیم فایل: python -m src.train_baselines
    main()
