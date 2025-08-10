# Stage App (1Y)

Streamlit application for Minervini-style market stage classification over the
most recent year.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate         # on Windows (or: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run stage_app/app.py
```

## Optional CLI

```bash
python -m stage_app.cli classify --ticker SPY --csv-out stages_1y.csv --suppress-warnings
```
