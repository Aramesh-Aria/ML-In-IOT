"""
هدف این فایل: تعریف یک «شاخص انرژی نسبی (Proxy)» برای مقایسه تصمیم‌های TPC.

نکته مهم:
- این مدل، انرژی واقعی سخت‌افزار یا باتری را شبیه‌سازی نمی‌کند.
- هدف، ساخت یک معیار ساده و قابل ارائه است تا نشان دهیم تغییر TP و SF چگونه
  می‌تواند به کاهش انرژی/مصرف منجر شود.

منطق کلی Proxy:
- توان ارسال بالاتر => مصرف انرژی بیشتر (تقریباً نمایی نسبت به dBm)
- SF بالاتر => Time-on-Air بیشتر => مصرف انرژی بیشتر (تقریباً نمایی نسبت به SF)
"""

from __future__ import annotations


def relative_energy(tp_dbm: float, sf: int) -> float:
    """
    محاسبه «انرژی نسبی» به عنوان یک Proxy ساده برای ارائه.

    ایده:
    - توان ارسال (Transmit Power) در مقیاس dBm تعریف می‌شود.
      تبدیل تقریبی dBm به مقیاس خطی: 10^(TP/10)
    - Spreading Factor (SF) در LoRa اگر افزایش یابد، مدت زمان ارسال (Time-on-Air)
      به شکل تقریبی نمایی افزایش می‌یابد (تقریباً proportional به 2^SF)

    بنابراین یک مدل ساده و قابل ارائه:
        RelativeEnergy ∝ 10^(TP/10) * 2^SF

    ورودی‌ها:
    - tp_dbm: توان ارسال بر حسب dBm
    - sf: Spreading Factor (عدد صحیح، معمولاً بین 7 تا 12)

    خروجی:
    - یک عدد مثبت که فقط «برای مقایسه» استفاده می‌شود (واحد فیزیکی واقعی ندارد).
      مقدار بزرگ‌تر یعنی انرژی نسبی بیشتر.
    """
    return (10 ** (tp_dbm / 10.0)) * (2 ** int(sf))


def normalized_energy(tp_dbm: float, sf: int, tp_ref: float = 14.0, sf_ref: int = 12) -> float:
    """
    نرمال‌سازی انرژی نسبت به یک حالت baseline.

    چرا نرمال می‌کنیم؟
    - چون relative_energy یک عدد بدون واحد است، برای مقایسه بهتر است آن را نسبت به
      یک نقطه مرجع (baseline) بیان کنیم.

    تعریف:
        energy_norm = relative_energy(tp_dbm, sf) / relative_energy(tp_ref, sf_ref)

    تفسیر خروجی:
    - اگر energy_norm < 1  => تصمیم TPC انرژی کمتری از baseline دارد (بهبود)
    - اگر energy_norm = 1  => برابر baseline
    - اگر energy_norm > 1  => بدتر از baseline (مصرف بیشتر)

    ورودی‌ها:
    - tp_dbm, sf: تصمیم فعلی TPC
    - tp_ref, sf_ref: baseline (پیش‌فرض 14 و 12)

    خروجی:
    - انرژی نرمال‌شده (یک عدد مثبت و قابل مقایسه)
    """
    return relative_energy(tp_dbm, sf) / relative_energy(tp_ref, sf_ref)
