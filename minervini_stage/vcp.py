"""Detection of Volatility Contraction Pattern (VCP) bases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VCPConfig:
    min_base_len: int = 30
    max_base_len: int = 65
    lookback: int = 3
    alpha_contract: float = 0.8
    min_contractions: int = 2
    dryup_beta: float = 0.7
    pivot_eps: float = 0.001


import pandas as pd

def _find_swings(close: pd.Series, lookback: int) -> pd.DataFrame:
    highs = []
    lows = []
    idx = close.index
    for i in range(lookback, len(close) - lookback):
        window = close.iloc[i - lookback : i + lookback + 1]
        price = close.iloc[i]
        if price == window.max():
            highs.append((idx[i], price))
        if price == window.min():
            lows.append((idx[i], price))
    highs_df = pd.DataFrame(highs, columns=["Date", "price"]).assign(type="H").set_index("Date")
    lows_df = pd.DataFrame(lows, columns=["Date", "price"]).assign(type="L").set_index("Date")
    swings = pd.concat([highs_df, lows_df]).sort_index()
    return swings
def detect_vcp(df: pd.DataFrame, config: VCPConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect VCP bases and pivot breakouts.

    Returns
    -------
    daily_df : DataFrame
        Original df with additional columns ``is_in_VCP_base``, ``is_VCP_pivot_breakout``, ``vcp_base_id``.
    bases : DataFrame
        Summary of detected bases.
    """
    cfg = config or VCPConfig()
    out = df.copy()
    out["is_in_VCP_base"] = False
    out["is_VCP_pivot_breakout"] = False
    out["vcp_base_id"] = pd.NA

    bases: List[dict] = []
    base_id = 0
    n = len(df)
    for start in range(0, n - cfg.min_base_len + 1):
        for length in range(cfg.min_base_len, cfg.max_base_len + 1):
            end = start + length
            if end >= n:
                break
            window = df.iloc[start:end]
            swings = _find_swings(window["Close"], cfg.lookback)
            if len(swings) < 4:
                continue
            ranges = swings["price"].diff().abs().dropna()
            if len(ranges) < cfg.min_contractions:
                continue
            contraction = all(
                ranges.iloc[i] < cfg.alpha_contract * ranges.iloc[i - 1]
                for i in range(1, len(ranges))
            )
            if not contraction:
                continue
            vol_early = window["Volume"].iloc[: length // 2].mean()
            vol_late = window["Volume"].iloc[length // 2 :].mean()
            if vol_late > cfg.dryup_beta * vol_early:
                continue
            pivot = swings[swings["type"] == "H"]["price"].iloc[-1]
            quality = (window["Close"] > window.get("SMA50", window["Close"]).fillna(0)).mean()
            bases.append(
                {
                    "base_id": base_id,
                    "base_start": window.index[0],
                    "base_end": window.index[-1],
                    "pivot_price": pivot,
                    "n_contractions": len(ranges),
                    "dryup_ratio": vol_late / vol_early if vol_early else np.nan,
                    "quality_score": float(quality),
                }
            )
            out.loc[window.index, "is_in_VCP_base"] = True
            out.loc[window.index, "vcp_base_id"] = base_id
            # mark breakout
            breakout_idx = df.index[end:][df["Close"].iloc[end:] > pivot * (1 + cfg.pivot_eps)]
            if len(breakout_idx) > 0:
                out.loc[breakout_idx[0], "is_VCP_pivot_breakout"] = True
            base_id += 1
    bases_df = pd.DataFrame(bases)
    return out, bases_df
