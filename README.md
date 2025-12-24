# Energy Optimization IoT (LoRaWAN + ML)

## نمای کلی

این پروژه یک پایپ‌لاین «انتها-به-انتها» برای **کاهش مصرف انرژی در LoRaWAN** ارائه می‌کند که در آن:

1. با **یادگیری ماشین** مقدار **SNR** پیش‌بینی می‌شود،
2. سپس با یک منطق **Transmission Power Control (TPC)**، مقادیر **Spreading Factor (SF)** و **Transmit Power (TP)** انتخاب می‌شوند تا ضمن حفظ حاشیه لینک، انرژی کاهش یابد. 

پایه مفهومی پروژه از ایده‌ی **Margin Excess (Me)** در ADR/TPC و نقش **Link Margin (LM)** و وابستگی آستانه **SNRlimit** به SF الهام گرفته شده است.

---

## ایده اصلی (Core Idea)

### 1) پیش‌بینی SNR

مدل ML با هدف `snr` آموزش داده می‌شود (پس ستون هدف باید در دیتاست موجود باشد). 

### 2) تصمیم‌گیری TPC با Margin (Me)

تصمیم‌ها بر اساس مفهوم margin ساخته می‌شوند (نسخه ساده‌شده و ارائه‌محور):

* اگر **Me < 0** → لینک از نظر شرط ما ایمن نیست ⇒ ابتدا **SF** زیاد می‌شود (تا SF_MAX) و در صورت نیاز **TP** زیاد می‌شود (تا TP_MAX).
* اگر **Me ≥ 0** → حاشیه داریم ⇒ ابتدا **SF** کم می‌شود و سپس **TP** کم می‌شود تا انرژی کاهش یابد، تا جایی که Me غیرمنفی بماند.

پارامترها و محدوده‌ها از `config.py` می‌آیند (SF: 7..12، TP: 2..14 dBm، و LM پیش‌فرض 10 dB).

### 3) شاخص انرژی (Proxy)

برای مقایسه ساده و قابل ارائه، یک **انرژی نسبی** استفاده می‌شود:

* `RE ∝ 10^(TP/10) * 2^SF`
* و سپس نسبت به baseline نرمال می‌شود (`energy_norm`).

Baseline پیش‌فرض برای مقایسه: `SF=12` و `TP=14`.

---

## ساختار خروجی‌ها (Artifacts)

پس از اجرای پایپ‌لاین:

* `outputs/predictions/snr_predictions.csv` شامل `snr_true` و `snr_pred`
* `outputs/predictions/tpc_decisions.csv` شامل `sf_new, tp_new, me, energy_norm`
* `outputs/figures/*.png` شامل نمودارهای:

  * `snr_true_vs_pred.png`
  * `sf_distribution.png`
  * `tp_distribution.png`
  * `me_distribution.png`
  * `energy_norm_hist.png` 

برای خلاصه‌سازی KPIها از فایل تصمیم‌ها:

* میانگین/میانه `energy_norm`
* درصد `energy_norm < 1` و `energy_norm < 0.5`
* درصد `Me >= 0` و آمارهای پرتکرار SF/TP 

---

## اسکریپت‌های اصلی

* `src/sanity_check.py` (اختیاری): بررسی سریع سلامت دیتاست (shape، ستون‌ها، NaN و …) 
* `src/train_baselines.py`: آموزش و مقایسه Ridge / RF / SVR و ذخیره مدل‌ها و متریک‌ها (RMSE و R²) 
* `src/run_pipeline.py`: اجرای End-to-End (پیش‌بینی SNR → تصمیم TPC → خروجی CSV/شکل‌ها) 
* `src/summarize_results.py`: استخراج KPIهای نهایی از `tpc_decisions.csv` 

---

## اجرای سریع (Quickstart)

### 1) نصب وابستگی‌ها

حداقل نیازها: `pandas, numpy, scikit-learn, matplotlib, joblib`

### 2) (اختیاری) چک دیتاست

```bash
python -m src.sanity_check
```

### 3) آموزش مدل‌ها و انتخاب مدل منتخب

```bash
python -m src.train_baselines
```

خروجی متریک‌ها در:

* `outputs/predictions/model_metrics.csv` 

### 4) اجرای پایپ‌لاین اصلی

```bash
python -m src.run_pipeline
```

خروجی‌ها در:

* `outputs/predictions/` و `outputs/figures/` 

### 5) خلاصه KPIها

```bash
python -m src.summarize_results
```

این اسکریپت KPIهای کلیدی (انرژی، SF/TP، margin) را چاپ می‌کند. 

---

## تنظیمات مهم (Configuration)

در `src/config.py` قابل تغییر است:

* مسیرها (data/models/outputs)
* ستون هدف (`snr`) و ستون‌های حذف‌شونده
* مدل منتخب (`SELECTED_TRAINED_MODEL`)
* محدوده SF/TP و مقدار `LINK_MARGIN_DB`
* baseline انرژی (`BASELINE_SF`, `BASELINE_TP`)

---

## نکات و محدودیت‌ها

* الگوریتم TPC و انرژی، **نمایشی/Proxy** هستند و هدفشان ارائه‌ی روند تصمیم‌گیری و مقایسه سناریوهاست (نه شبیه‌سازی دقیق فیزیک لینک یا باتری).
* اگر بخواهید پروژه به نسخه «واقع‌گرایانه‌تر» نزدیک شود، مسیر طبیعی توسعه این است که:

  * مدل انرژی را به ToA و پارامترهای PHY دقیق‌تر متصل کنید،
  * و KPIهای شبکه مثل PDR/Collision را نیز وارد ارزیابی کنید (مطابق ادبیات ML/RL در تخصیص منابع LoRaWAN).

---

## رفرنس‌های پژوهشی (پایه ایده)

* ML-Assisted TPC با محوریت SNR prediction و مفهوم Me/LM

* Machine-Learning-Assisted Transmission Power
Control for LoRaWAN Considering
Environments With High Signal-
to-Noise Variation (MAURICIO GONZÁLEZ-PALACIO , DIANA TOBÓN-VALLEJO, LINA M. SEPÚLVEDA-CANO
,
MARIO LUNA-DELRISCO, CHRISTOF RÖEHRIG , (Member, IEEE),
AND LONG BAO LE ,(Fellow, IEEE))