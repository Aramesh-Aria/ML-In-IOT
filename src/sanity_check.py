"""
هدف این اسکریپت:
- یک بررسی سریع و سبک (Sanity Check) روی دیتاست خام انجام می‌دهد تا قبل از اجرای آموزش/پایپ‌لاین،
  از سالم بودن داده مطمئن شوید.

چه چیزهایی را چک می‌کند؟
1) Shape: تعداد سطرها و ستون‌ها
2) Columns: لیست ستون‌ها (برای اطمینان از وجود ستون هدف مثل snr)
3) Head: چند ردیف اول برای دیدن نمونه داده‌ها
4) Missing values: بررسی مقدارهای گمشده (NaN) در ستون‌ها
5) Describe: آمار توصیفی اولیه برای تشخیص داده‌های غیرعادی (مثلاً min/max عجیب)

این فایل معمولاً قبل از train_baselines یا run_pipeline اجرا می‌شود.
"""

from pathlib import Path
import pandas as pd


# مسیر پیش‌فرض دیتاست خام
# نکته: این مسیر نسبت به «ریشه پروژه» تعریف شده است.
# اگر این اسکریپت از جای دیگری اجرا شود، ممکن است مسیر پیدا نشود.
DATA_PATH_DEFAULT = Path("data/raw/subsampled_data.csv")


def load_dataset(path: Path) -> pd.DataFrame:
    """
    بارگذاری دیتاست از مسیر داده‌شده.

    ورودی:
    - path: مسیر فایل CSV

    خروجی:
    - DataFrame پانداس
    """
    return pd.read_csv(path)


def report_basic_stats(df: pd.DataFrame) -> None:
    """
    چاپ گزارش آماری اولیه برای دیتاست.

    خروجی‌ها:
    - Shape و ستون‌ها: برای تأیید ساختار کلی داده
    - Head: چند ردیف اول برای بررسی سریع کیفیت داده
    - Missing values: شمارش NaNها (اگر زیاد باشد باید قبل از مدل‌سازی حل شود)
    - Describe: آمار توصیفی (میانگین/انحراف معیار/min/max و ...) برای تشخیص مقادیر غیرعادی
    """
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # نمایش چند ردیف اول برای اطمینان از درست خوانده شدن CSV
    print("\nHead:\n", df.head())

    # بررسی مقدارهای گمشده: اگر ستونی NaN زیاد داشته باشد ممکن است مدل‌سازی را خراب کند
    print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False).head(20))

    # آمار توصیفی کلی (برای 20 ستون اول) — کمک می‌کند outlier یا مقیاس غیرعادی را سریع ببینیم
    print("\nDescribe (first 20 rows):\n", df.describe(include="all").T.head(20))


def main():
    """
    نقطه شروع اجرای اسکریپت.

    مراحل:
    1) خواندن دیتاست خام از مسیر پیش‌فرض
    2) چاپ گزارش آماری اولیه
    """
    df = load_dataset(DATA_PATH_DEFAULT)
    report_basic_stats(df)


if __name__ == "__main__":
    # اگر با python -m src.sanity_check یا python sanity_check.py اجرا شود،
    # این بخش main را اجرا می‌کند.
    main()
