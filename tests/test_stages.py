import json
from pathlib import Path
from datetime import datetime, UTC
import pandas as pd
import pytest

# Try importing from either stage or stage_app.stage
try:
    from stage import fetch_price_data, compute_indicators, classify_stages  # type: ignore
except Exception:  # pragma: no cover
    from stage_app.stage import fetch_price_data, compute_indicators, classify_stages

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = BASE_DIR / "golden_windows.yaml"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

with GOLDEN_PATH.open() as f:
    GOLDEN_WINDOWS = json.load(f)


def _save_debug(df: pd.DataFrame, name: str, columns: list[str]) -> Path:
    path = ARTIFACTS_DIR / f"{name}_debug.csv"
    df[columns].to_csv(path)
    return path


def _run_continuity(cfg: dict) -> None:
    years = cfg["years"]
    exp = cfg["expect"]
    lookback = years * 365 + 400
    df = fetch_price_data(cfg["ticker"], lookback_days=lookback)
    df = compute_indicators(df)
    df["Stage"] = classify_stages(df)
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=years * 365)
    recent = df.loc[start_date:]

    if len(recent) < exp["min_trading_days"]:
        path = _save_debug(recent, cfg["name"], ["Open","High","Low","Close","SMA25","SMA50","SMA150","SMA200","Stage"])
        pytest.fail(
            f"{len(recent)} trading days found (<{exp['min_trading_days']}). Debug CSV: {path}"
        )

    coverage = recent.tail(exp["require_sma200_coverage_days"])
    if coverage["SMA200"].isna().any():
        missing = int(coverage["SMA200"].isna().sum())
        path = _save_debug(recent, cfg["name"], ["Open","High","Low","Close","SMA25","SMA50","SMA150","SMA200","Stage"])
        pytest.fail(
            f"SMA200 has {missing} NaNs in last {exp['require_sma200_coverage_days']} days. Debug CSV: {path}"
        )

    nan_ratio = float(recent["Stage"].isna().mean())
    if nan_ratio > exp["allow_stage_nan_ratio"]:
        path = _save_debug(recent, cfg["name"], ["Open","High","Low","Close","SMA25","SMA50","SMA150","SMA200","Stage"])
        pytest.fail(
            f"Stage NaN ratio {nan_ratio:.2%} exceeds {exp['allow_stage_nan_ratio']:.2%}. Debug CSV: {path}"
        )


def _run_stage_window(cfg: dict) -> None:
    window = cfg["window"]
    exp = cfg["expect"]
    start = pd.to_datetime(window["start"])
    end = pd.to_datetime(window["end"])
    utc_today = datetime.now(UTC).date()
    lookback = (utc_today - start.date()).days + 400
    df = fetch_price_data(cfg["ticker"], lookback_days=lookback)
    df = compute_indicators(df)
    df["Stage"] = classify_stages(df)
    window_df = df.loc[start:end]

    counts = window_df["Stage"].dropna().astype(int).value_counts()
    if counts.empty:
        path = _save_debug(window_df, cfg["name"], ["Close","SMA200","Slope200","Stage"])
        pytest.fail(f"No stage data in window. Debug CSV: {path}")

    most_stage = int(counts.idxmax())
    stage_ratio = counts.get(exp["stage"], 0) / counts.sum()
    if most_stage != exp["stage"] or stage_ratio < exp["min_ratio"]:
        path = _save_debug(window_df, cfg["name"], ["Close","SMA200","Slope200","Stage"])
        pytest.fail(
            "most_stage={ms}, stage{st}_ratio={r:.2%}, counts={c}. Debug CSV: {p}".format(
                ms=most_stage, st=exp["stage"], r=stage_ratio, c=counts.to_dict(), p=path
            )
        )


@pytest.mark.parametrize("cfg", GOLDEN_WINDOWS, ids=[w["name"] for w in GOLDEN_WINDOWS])
def test_golden_windows(cfg: dict) -> None:
    if "years" in cfg:
        _run_continuity(cfg)
    else:
        _run_stage_window(cfg)


def test_slope_smooth_window() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {
            "Close": [110] * 5,
            "SMA50": [100] * 5,
            "SMA150": [100] * 5,
            "SMA200": [100] * 5,
            "Slope200": [3, 3, 3, -4, -4],
            "High52w": [120] * 5,
            "Low52w": [90] * 5,
        },
        index=idx,
    )
    default_stage = classify_stages(df).iloc[-1]
    custom_stage = classify_stages(df, slope_smooth_window=3).iloc[-1]
    assert default_stage == 2
    assert custom_stage == 3


def test_below200_margin_triggers_stage4() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {
            "Close": [80] * 5,
            "SMA50": [100] * 5,
            "SMA150": [100] * 5,
            "SMA200": [100] * 5,
            "Slope200": [0.1] * 5,
            "High52w": [120] * 5,
            "Low52w": [60] * 5,
        },
        index=idx,
    )
    stage_margin = classify_stages(df).iloc[-1]
    stage_no_margin = classify_stages(df, below200_margin=0).iloc[-1]
    assert stage_margin == 4
    assert stage_no_margin == 1
