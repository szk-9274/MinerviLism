# Stage App (1Y)

Streamlit application for Minervini-style market stage classification over the
most recent year.

## What it does

The `stage_app` package downloads daily OHLCV data via `yfinance`, computes
common moving averages (25/50/150/200), 52‑week high/low and the slope of the
200‑day average, and finally labels each day with one of Mark Minervini's
four market stages:

| Stage | Meaning (simplified) | Colour |
|-------|---------------------|--------|
| 1     | Below a flat/upward 200‑MA   | gray   |
| 2     | Above a rising 200‑MA        | green  |
| 3     | Above a falling 200‑MA       | orange |
| 4     | Below a falling 200‑MA       | red    |

See `stage_app/stage.py` for the exact rules.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate         # on Windows (or: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run stage_app/app.py
```

## Optional CLI

```bash
python -m stage_app.cli classify --ticker SPY --csv-out stages_1y.csv \
    --slope-smooth-window 5 --suppress-warnings
```

### Library use

The core functionality is available as a library for use in notebooks or other
scripts:

```python
from stage_app.stage import fetch_price_data, compute_indicators, classify_stages

df = fetch_price_data("SPY", lookback_days=380)
df = compute_indicators(df)
# ``slope_smooth_window`` controls the rolling mean used for the 200MA slope.
df["Stage"] = classify_stages(df, slope_smooth_window=5)
```

`df` now contains the moving averages and a `Stage` column ready for analysis
or charting.
## Run tests
```bash
python -m venv venv
venv\Scripts\activate            # Windows (or: source venv/bin/activate)
pip install -r requirements.txt  # app deps
pip install -r requirements-dev.txt
pytest -q
```
