#جدول خلاصه توزیع SF/TP و درصد Me≥0
from sys import displayhook
import pandas as pd
from src import config
from src.tpc import snr_limit
from src.energy import normalized_energy

# خروجی‌های پایپ‌لاین
dec = pd.read_csv(config.TPC_DECISIONS_CSV)          # sf_new, tp_new, me, energy_norm
pred = pd.read_csv(config.SNR_PREDICTIONS_CSV)       # snr_true, snr_pred

# هم‌ردیف کردن (چون هر دو فایل به ترتیب یکسان ساخته شده‌اند)
assert len(dec) == len(pred), "Length mismatch between predictions and decisions!"
df = pd.concat([pred, dec], axis=1)

# -------- Baseline (قبل از TPC) --------
df["sf_old"] = config.BASELINE_SF
df["tp_old"] = float(config.BASELINE_TP)

# تعریف Margin برای baseline:
# Me_old = SNR_pred - SNR_limit(SF_old) - LM
df["me_old"] = df["snr_pred"] - snr_limit(df["sf_old"].iloc[0]) - config.LINK_MARGIN_DB

# انرژی baseline نسبت به خودش = 1
df["energy_norm_old"] = 1.0

# -------- نمایش مقایسه توزیع‌ها --------
print("SF counts (old vs new):")
sf_cmp = pd.DataFrame({
    "old": df["sf_old"].value_counts().sort_index(),
    "new": df["sf_new"].value_counts().sort_index(),
}).fillna(0).astype(int)
displayhook(sf_cmp)

print("TP counts (old vs new):")
tp_cmp = pd.DataFrame({
    "old": df["tp_old"].value_counts().sort_index(),
    "new": df["tp_new"].value_counts().sort_index(),
}).fillna(0).astype(int)
displayhook(tp_cmp)

# -------- درصد Margin >= 0 --------
pct_me_ge_0_old = (df["me_old"] >= 0).mean() * 100
pct_me_ge_0_new = (df["me"] >= 0).mean() * 100

print("pct_me_ge_0_old =", round(pct_me_ge_0_old, 2), "%")
print("pct_me_ge_0_new =", round(pct_me_ge_0_new, 2), "%")

# -------- خلاصه انرژی (کامل‌تر) --------
energy_mean_new = float(df["energy_norm"].mean())
energy_median_new = float(df["energy_norm"].median())

pct_energy_below_1_new = float((df["energy_norm"] < 1.0).mean() * 100)
pct_energy_below_0_5_new = float((df["energy_norm"] < 0.5).mean() * 100)

# کاهش نسبت به baseline=1.0
mean_reduction_pct = (1.0 - energy_mean_new) * 100
median_reduction_pct = (1.0 - energy_median_new) * 100

print("\n--- Energy summary ---")
print("energy_norm_old mean =", float(df["energy_norm_old"].mean()))  # همیشه 1.0
print("energy_norm_new mean =", round(energy_mean_new, 4))
print("energy_norm_new median =", round(energy_median_new, 4))

print("pct_energy_below_1_new =", round(pct_energy_below_1_new, 2), "%")
print("pct_energy_below_0_5_new =", round(pct_energy_below_0_5_new, 2), "%")

print("mean_energy_reduction_vs_baseline =", round(mean_reduction_pct, 2), "%")
print("median_energy_reduction_vs_baseline =", round(median_reduction_pct, 2), "%")
