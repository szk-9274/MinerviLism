"""Stage classification logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StageConfig:
    """Configuration for stage classification."""
    slope_threshold: float = 0.0
    beta_norm_threshold: Optional[float] = None


def classify_stage(df: pd.DataFrame, config: StageConfig | None = None) -> pd.DataFrame:
    """Classify market stages according to Minervini rules.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Close``, ``SMA50``, ``SMA150``, ``SMA200``,
        ``SMA200_slope``, ``High52w``, ``Low52w``.
    config : StageConfig
        Threshold parameters.
    """
    cfg = config or StageConfig()
    out = df.copy()
    stage = pd.Series(index=df.index, dtype="float")

    tt = pd.Series(True, index=df.index)
    tt &= out["Close"] > out["SMA50"]
    tt &= out["Close"] > out["SMA150"]
    tt &= out["Close"] > out["SMA200"]
    tt &= out["SMA150"] > out["SMA200"]
    if cfg.beta_norm_threshold is not None:
        tt &= out["SMA200_slope_norm"] > cfg.beta_norm_threshold
    else:
        tt &= out["SMA200_slope"] > cfg.slope_threshold
    tt &= out["Close"] >= out["Low52w"] * 1.30
    tt &= out["Close"] >= out["High52w"] * 0.75

    stage[tt] = 2

    stage[(out["Close"] < out["SMA200"]) & (out["SMA200_slope"] < 0) & ~tt] = 4

    cond3 = (
        (out["Close"] < out["SMA50"]) & (out["Close"] < out["SMA150"])
        & (out["Close"] >= out["SMA200"]) & (out["SMA200_slope"] <= 0)
    )
    stage[cond3 & stage.isna()] = 3

    cond1 = (out["Close"] <= out["SMA200"]) & (out["SMA200_slope"] >= 0)
    stage[cond1 & stage.isna()] = 1

    out["Stage"] = stage
    return out
