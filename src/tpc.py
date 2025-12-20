from __future__ import annotations
from dataclasses import dataclass
from . import config

@dataclass
class TPCDecision:
    sf: int
    tp: float
    me: float

def snr_limit(sf: int) -> float:
    return config.SNR_LIMIT_BY_SF[int(sf)]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def decide_tpc(snr_pred: float, sf_start: int | None = None, tp_start: float | None = None) -> TPCDecision:
    """
    Simplified ML-assisted TPC decision based on margin:
      Me = SNR_pred - SNR_limit(SF) - LM

    Strategy (presentation-friendly):
      - If Me > 0: decrease SF until Me <= 0 or SF hits SF_MIN (reduce ToA)
      - If Me < 0: increase SF (improve robustness) up to SF_MAX
                 and if still negative, increase TP to compensate (up to TP_MAX)
      - Output final (SF, TP, Me)
    """
    sf = int(sf_start) if sf_start is not None else config.BASELINE_SF
    tp = float(tp_start) if tp_start is not None else float(config.BASELINE_TP)

    def me(sf_val: int, tp_val: float) -> float:
        snr_eff = snr_pred + (tp_val - config.BASELINE_TP)
        return snr_eff - snr_limit(sf_val) - config.LINK_MARGIN_DB

    # اگر margin منفی است: ابتدا SF را بالا ببر، بعد TP را بالا ببر
    while me(sf, tp) < 0 and sf < config.SF_MAX:
        sf += 1

    while me(sf, tp) < 0 and tp < config.TP_MAX:
        tp += 1.0

    # اگر margin مثبت است: ابتدا SF را پایین بیاور، سپس TP را پایین بیاور
    while sf > config.SF_MIN and me(sf - 1, tp) >= 0:
        sf -= 1

    while tp > config.TP_MIN and me(sf, tp - 1.0) >= 0:
        tp -= 1.0

    return TPCDecision(sf=sf, tp=tp, me=float(me(sf, tp)))