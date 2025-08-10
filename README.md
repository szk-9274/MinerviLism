# Minervini Stage Module

Utilities for classifying market stages and detecting Minervini Volatility Contraction Patterns (VCP).

## Quick start

```bash
pip install -e .
python -m minervini_stage.cli classify --ticker SPY --years 10 --out stages.csv
python -m minervini_stage.cli vcp --ticker AAPL --years 5 --json vcp.json
python -m minervini_stage.cli plot --ticker TSLA --years 5 --out chart.png --with-vcp
```

Run Streamlit viewer:

```bash
streamlit run minervini_stage/app_streamlit.py
```

## Parameters

All thresholds are exposed via configuration classes (`StageConfig`, `VCPConfig`, `IndicatorConfig`).

## Testing

```bash
pytest -q
```
