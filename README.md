
# 10-Year Stock Replay & Savings Overlay

Supports both English and Japanese interfaces. Use the sidebar to switch languages.
You can also toggle between desktop and mobile layouts from the sidebar.

The ticker selector includes popular U.S. stocks and Bitcoin by default, and you can
enter any other Yahoo Finance symbol manually. Leaving the ticker blank shows a
friendly warning instead of an error.

The savings model now renders a visible blue line starting from zero, making it
easy to compare contributions against price performance.

## Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```
