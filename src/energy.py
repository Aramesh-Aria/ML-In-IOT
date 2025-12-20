from __future__ import annotations
import math

def relative_energy(tp_dbm: float, sf: int) -> float:
    """
    Presentation-friendly relative energy proxy:
      RE ∝ 10^(TP/10) * 2^SF
    """
    return (10 ** (tp_dbm / 10.0)) * (2 ** int(sf))

def normalized_energy(tp_dbm: float, sf: int, tp_ref: float = 14.0, sf_ref: int = 12) -> float:
    return relative_energy(tp_dbm, sf) / relative_energy(tp_ref, sf_ref)
