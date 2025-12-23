"""
هدف این اسکریپت:
- خواندن خروجی تصمیم‌های TPC از فایل outputs/predictions/tpc_decisions.csv
- محاسبه KPIهای خلاصه و قابل ارائه برای گزارش نهایی

این KPIها به شما کمک می‌کنند سریع پاسخ دهید:
- آیا انرژی (proxy) نسبت به baseline کاهش یافته؟
- TPC بیشتر چه SF/TPهایی را انتخاب کرده؟
- Margin (Me) بعد از تصمیم‌گیری چقدر «ایمن» بوده (چند درصد Me>=0)؟

این فایل معمولاً بعد از run_pipeline اجرا می‌شود.
"""

import pandas as pd
from src import config


def top_counts(series: pd.Series, n: int = 5) -> dict:
    """
    استخراج پرتکرارترین مقادیر یک سری (برای خلاصه‌سازی در گزارش).

    ورودی:
    - series: ستون مورد نظر (مثلاً sf_new یا tp_new)
    - n: تعداد آیتم‌های پرتکرار که می‌خواهیم نگه داریم

    خروجی:
    - یک دیکشنری از نوع:
      {"value1": count1, "value2": count2, ...}
      تبدیل کلیدها به str برای چاپ/ذخیره تمیزتر انجام می‌شود.
    """
    vc = series.value_counts().head(n)
    return {str(k): int(v) for k, v in vc.items()}


def main():
    """
    اجرای اصلی استخراج نتایج خلاصه.

    مراحل:
    1) خواندن فایل tpc_decisions.csv
    2) کنترل اینکه ستون‌های ضروری موجود باشند
    3) محاسبه KPIهای انرژی، SF، TP و Margin
    4) رُند کردن خروجی برای چاپ خواناتر
    5) چاپ دیکشنری نهایی
    """
    # -------------------------------------------------------------------------
    # 1) Load decisions file created by run_pipeline.py
    # -------------------------------------------------------------------------
    dec = pd.read_csv(config.TPC_DECISIONS_CSV)

    # -------------------------------------------------------------------------
    # 2) Basic schema sanity check
    # اگر این ستون‌ها نباشند یعنی run_pipeline درست تولید نکرده یا فایل اشتباه است
    # -------------------------------------------------------------------------
    required = {"sf_new", "tp_new", "me", "energy_norm"}
    missing = required - set(dec.columns)
    if missing:
        raise ValueError(f"Missing required columns in tpc_decisions.csv: {missing}")

    # -------------------------------------------------------------------------
    # 3) Compute KPIs (Key Performance Indicators)
    # -------------------------------------------------------------------------
    summary = {
        "count": int(len(dec)),

        # -----------------------------
        # Energy (Proxy) KPIs
        # -----------------------------
        # میانگین energy_norm (نسبت به baseline)
        "energy_norm_mean": float(dec["energy_norm"].mean()),

        # میانه energy_norm (گاهی از میانگین مقاوم‌تر است، مخصوصاً اگر outlier داشته باشیم)
        "energy_norm_median": float(dec["energy_norm"].median()),

        # درصد نمونه‌هایی که energy_norm < 1 => بهتر از baseline
        "pct_energy_below_1": float((dec["energy_norm"] < 1.0).mean() * 100),

        # درصد نمونه‌هایی که energy_norm < 0.5 => حداقل 50% بهتر از baseline (proxy)
        "pct_energy_below_0_5": float((dec["energy_norm"] < 0.5).mean() * 100),

        # -----------------------------
        # SF KPIs
        # -----------------------------
        # پرتکرارترین SF انتخابی (Mode)
        "sf_mode": int(dec["sf_new"].mode().iloc[0]),

        # حداقل و حداکثر SF انتخاب‌شده
        "sf_min": int(dec["sf_new"].min()),
        "sf_max": int(dec["sf_new"].max()),

        # پرتکرارترین SFها و تعدادشان (برای تحلیل رفتار الگوریتم)
        "sf_top_counts": top_counts(dec["sf_new"], n=6),

        # -----------------------------
        # TP KPIs
        # -----------------------------
        "tp_mean": float(dec["tp_new"].mean()),
        "tp_median": float(dec["tp_new"].median()),
        "tp_min": float(dec["tp_new"].min()),
        "tp_max": float(dec["tp_new"].max()),

        # چون tp_new ممکن است float باشد، برای جمع‌بندی بهتر round می‌کنیم
        # (در پروژه شما معمولاً گام 1 dBm است، پس round(1) کافی است)
        "tp_top_counts": top_counts(dec["tp_new"].round(1), n=8),

        # -----------------------------
        # Margin (Me) KPIs
        # -----------------------------
        "me_mean": float(dec["me"].mean()),
        "me_median": float(dec["me"].median()),

        # درصد نمونه‌هایی که margin غیرمنفی است (یعنی لینک از نظر شرط ما “ایمن/قابل قبول” است)
        "pct_me_ge_0": float((dec["me"] >= 0.0).mean() * 100),
    }

    # -------------------------------------------------------------------------
    # 4) Optional: round floats for nicer printing
    # این بخش فقط خروجی چاپی را تمیز می‌کند و روی محاسبات اثری ندارد.
    # -------------------------------------------------------------------------
    pretty = {}
    for k, v in summary.items():
        if isinstance(v, float):
            pretty[k] = round(v, 4)
        else:
            pretty[k] = v

    # -------------------------------------------------------------------------
    # 5) Print final summary
    # -------------------------------------------------------------------------
    print(pretty)


if __name__ == "__main__":
    # اجرای مستقیم: python -m src.summarize_results
    main()
