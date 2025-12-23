"""
این فایل «اجرای انتها-به-انتها (End-to-End)» پروژه را انجام می‌دهد.

ورودی:
- data/raw/subsampled_data.csv  (یا processed اگر بعداً اضافه شود)
- مدل آموزش‌داده‌شده در models_trained/ (مثلاً ridge.joblib)

خروجی:
1) outputs/predictions/snr_predictions.csv
   شامل: snr_true و snr_pred

2) outputs/predictions/tpc_decisions.csv
   شامل: sf_new, tp_new, me, energy_norm
   که sf_new و tp_new تصمیم TPC بر اساس SNR پیش‌بینی‌شده هستند.

3) outputs/figures/*.png
   نمودارهای کلیدی برای گزارش و ارائه:
   - snr_true_vs_pred.png
   - sf_distribution.png
   - tp_distribution.png
   - me_distribution.png
   - energy_norm_hist.png

فلسفه کلی:
- ابتدا SNR را با مدل ML پیش‌بینی می‌کنیم
- سپس بر اساس SNR پیش‌بینی‌شده، تصمیم‌های TPC (SF/TP) را استخراج می‌کنیم
- در نهایت برای تحلیل، نمودار و فایل خروجی ذخیره می‌کنیم
"""

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
    """
    اجرای کامل پایپ‌لاین پروژه.

    مراحل:
    1) ساخت پوشه‌های خروجی (اگر وجود ندارند)
    2) خواندن دیتاست (در این نسخه از raw استفاده می‌شود)
    3) حذف ستون‌های غیرمفید برای ML (DROP_COLS)
    4) جداسازی X و y (هدف: snr)
    5) لود مدل منتخب (SELECTED_TRAINED_MODEL از models_trained)
    6) پیش‌بینی SNR برای همه نمونه‌ها
    7) ذخیره CSV پیش‌بینی‌ها
    8) اجرای الگوریتم تصمیم‌گیری TPC برای هر نمونه:
       - خروجی: sf_new, tp_new, me و energy_norm
    9) ذخیره CSV تصمیم‌ها
    10) تولید نمودارهای گزارش (برای ارائه)
    """
    # -------------------------------------------------------------------------
    # 1) Ensure output directories exist
    # -------------------------------------------------------------------------
    ensure_dirs()

    # -------------------------------------------------------------------------
    # 2) Load dataset
    # prefer_processed=False یعنی از دیتای خام استفاده کن (چون processed فعلاً نداریم/لازم نیست)
    # copy() برای جلوگیری از تغییر ناخواسته روی df اصلی
    # -------------------------------------------------------------------------
    df = load_dataset(prefer_processed=False).copy()

    # -------------------------------------------------------------------------
    # 3) Drop non-ML columns (ستون‌های شناسه‌ای/زمانی/متنی که برای ML مناسب نیستند)
    # این ستون‌ها در config.DROP_COLS تعریف شده‌اند.
    # -------------------------------------------------------------------------
    drop_cols = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # -------------------------------------------------------------------------
    # 4) Detect target column and split to X/y
    # target معمولاً "snr" است (طبق config.TARGET_COL)
    # -------------------------------------------------------------------------
    target = detect_target_col(df)

    # X = همه ستون‌ها به جز ستون هدف
    X = df.drop(columns=[target]).copy()

    # y_true = SNR واقعی برای مقایسه با پیش‌بینی مدل
    y_true = df[target].copy()

    # -------------------------------------------------------------------------
    # 5) Load trained model (مدل آموزش‌داده‌شده توسط خودمان)
    # مدل منتخب از config.SELECTED_TRAINED_MODEL می‌آید (مثلاً ridge.joblib)
    # -------------------------------------------------------------------------
    model_path = config.TRAINED_MODELS_DIR / config.SELECTED_TRAINED_MODEL
    model = joblib.load(model_path)

    # -------------------------------------------------------------------------
    # 6) Make sure X is numeric and predict SNR
    # safe_numeric_X ستون‌های غیرعددی را به عدد تبدیل می‌کند و NaNها را پر می‌کند
    # -------------------------------------------------------------------------
    Xn = safe_numeric_X(X)
    snr_pred = model.predict(Xn)

    # -------------------------------------------------------------------------
    # 7) Save predictions to CSV
    # این فایل پایه تحلیل‌هاست: نمودار snr_true_vs_pred از همینجا تولید می‌شود
    # -------------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "snr_true": y_true.values,
        "snr_pred": snr_pred,
    })
    save_csv(pred_df, config.SNR_PREDICTIONS_CSV)

    # -------------------------------------------------------------------------
    # 8) Run TPC decisions per sample (بر اساس SNR پیش‌بینی‌شده)
    # برای هر snr_pred:
    #  - decide_tpc یک تصمیم (sf,tp) و margin Me می‌دهد
    #  - سپس energy_norm را نسبت به baseline محاسبه می‌کنیم
    # -------------------------------------------------------------------------
    decisions = []
    for sp in snr_pred:
        d = decide_tpc(float(sp))

        decisions.append({
            "sf_new": d.sf,   # SF انتخابی TPC
            "tp_new": d.tp,   # TP انتخابی TPC (dBm)
            "me": d.me,       # Margin after decision (Me)
            # انرژی نرمال‌شده نسبت به baseline (SF=12, TP=14)
            "energy_norm": normalized_energy(
                d.tp,
                d.sf,
                tp_ref=config.BASELINE_TP,
                sf_ref=config.BASELINE_SF
            ),
        })

    dec_df = pd.DataFrame(decisions)
    save_csv(dec_df, config.TPC_DECISIONS_CSV)

    # -------------------------------------------------------------------------
    # 9) Generate figures for report (نمودارهای ارائه)
    # این نمودارها خروجی‌های بصری اصلی پروژه هستند.
    # -------------------------------------------------------------------------

    # 9.1) True vs Pred: آیا مدل SNR را خوب پیش‌بینی کرده؟
    plt.figure()
    plt.scatter(pred_df["snr_true"], pred_df["snr_pred"])
    plt.xlabel("SNR true (dB)")
    plt.ylabel("SNR predicted (dB)")
    plt.title("SNR: True vs Predicted")
    plt.savefig(config.FIG_DIR / "snr_true_vs_pred.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 9.2) Distribution of chosen SF: TPC چه SFهایی را بیشتر انتخاب کرده؟
    plt.figure()
    dec_df["sf_new"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("SF chosen")
    plt.ylabel("Count")
    plt.title("TPC Output: SF Distribution")
    plt.savefig(config.FIG_DIR / "sf_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 9.3) Distribution of chosen TP: TPC چه TPهایی را بیشتر انتخاب کرده؟
    plt.figure()
    plt.hist(dec_df["tp_new"], bins=13)  # مناسب برای بازه 2..14 با گام 1
    plt.xlabel("TP chosen (dBm)")
    plt.ylabel("Count")
    plt.title("TPC Output: TP Distribution")
    plt.savefig(config.FIG_DIR / "tp_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 9.4) Margin histogram: وضعیت margin بعد از تصمیم‌گیری چطور است؟
    plt.figure()
    plt.hist(dec_df["me"], bins=20)
    plt.xlabel("Margin Me (dB)")
    plt.ylabel("Count")
    plt.title("Margin Distribution after TPC")
    plt.savefig(config.FIG_DIR / "me_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 9.5) Energy norm histogram: انرژی نسبی نسبت به baseline چگونه تغییر کرده؟
    plt.figure()
    plt.hist(dec_df["energy_norm"], bins=20)
    plt.xlabel("Normalized energy (vs baseline SF=12, TP=14)")
    plt.ylabel("Count")
    plt.title("Energy Reduction Proxy")
    plt.savefig(config.FIG_DIR / "energy_norm_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 10) Print outputs path for quick navigation
    # -------------------------------------------------------------------------
    print("Saved predictions:", config.SNR_PREDICTIONS_CSV)
    print("Saved decisions:", config.TPC_DECISIONS_CSV)
    print("Saved figures in:", config.FIG_DIR)


if __name__ == "__main__":
    # اگر این فایل با python -m src.run_pipeline اجرا شود، main فراخوانی می‌شود.
    main()
