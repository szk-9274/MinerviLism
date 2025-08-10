
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="10-Year Stock Replay & Savings Overlay", layout="wide")

LANG_OPTIONS = {"English": "en", "日本語": "ja"}
translations = {
    "en": {
        "app_title": "📈 10-Year Stock Replay & Savings Overlay",
        "settings": "Settings",
        "ticker_input": "Ticker (Yahoo Finance symbol)",
        "years_of_history": "Years of history",
        "end_date": "End date",
        "start_date": "Start date",
        "savings_model": "Savings model",
        "monthly_contrib": "Monthly contribution",
        "annual_interest": "Annual interest (%)",
        "normalize": "Normalize to 100 at start",
        "fps": "Playback speed (frames/sec)",
        "frame_granularity": "Frame granularity",
        "monthly": "Monthly",
        "quarterly": "Quarterly",
        "yearly": "Yearly",
        "no_price_data": "No price data found. Check the ticker or date range.",
        "price_change": "Price change",
        "total_contrib": "Total contributions",
        "savings_balance": "Savings balance",
        "play": "\u25b6 Play",
        "pause": "\u23f8 Pause",
        "up_to": "Up to: ",
        "indexed_title": "Indexed to 100",
        "price_savings_title": "Price / Savings",
        "replay_title": "{ticker} — {years}Y Replay with Savings Overlay",
        "date": "Date",
        "savings_label": "Savings"
    },
    "ja": {
        "app_title": "📈 10年間株価再生と貯蓄の重ね合わせ",
        "settings": "設定",
        "ticker_input": "ティッカー（Yahoo Financeのシンボル）",
        "years_of_history": "表示年数",
        "end_date": "終了日",
        "start_date": "開始日",
        "savings_model": "貯蓄モデル",
        "monthly_contrib": "毎月の積立額",
        "annual_interest": "年利（％）",
        "normalize": "開始時100で正規化",
        "fps": "再生速度（フレーム/秒）",
        "frame_granularity": "フレーム粒度",
        "monthly": "月次",
        "quarterly": "四半期",
        "yearly": "年次",
        "no_price_data": "価格データがありません。ティッカーや日付範囲を確認してください。",
        "price_change": "価格変化",
        "total_contrib": "総積立額",
        "savings_balance": "貯蓄残高",
        "play": "\u25b6 \u518d\u751f",
        "pause": "\u23f8 \u4e00\u6642\u505c\u6b62",
        "up_to": "\u3053\u3053\u307e\u3067: ",
        "indexed_title": "\u958b\u59cb\u6642\u3092100\u3068\u3057\u3066\u6307\u6570\u5316",
        "price_savings_title": "\u4fa1\u683c / \u8caf\u84c4",
        "replay_title": "{ticker} — {years}年リプレイと貯蓄のオーバーレイ",
        "date": "\u65e5\u4ed8",
        "savings_label": "\u8caf\u84c4"
    }
}

lang_display = st.sidebar.selectbox("Language / \u8a00\u8a9e", options=list(LANG_OPTIONS.keys()), index=0)
lang = LANG_OPTIONS[lang_display]

def t(key, **kwargs):
    return translations.get(lang, translations["en"]).get(key, translations["en"].get(key, key)).format(**kwargs)

st.title(t("app_title"))

with st.sidebar:
    st.header(t("settings"))
    ticker = st.text_input(t("ticker_input"), value="SPY")
    years = st.slider(t("years_of_history"), min_value=3, max_value=20, value=10, step=1)
    end_date = st.date_input(t("end_date"), value=date.today())
    start_date = st.date_input(t("start_date"), value=(end_date - timedelta(days=int(365.25*years))))
    st.markdown("---")
    st.subheader(t("savings_model"))
    monthly_contrib = st.number_input(t("monthly_contrib"), min_value=0.0, value=30000.0, step=1000.0, format="%.0f")
    annual_interest = st.number_input(t("annual_interest"), min_value=0.0, max_value=20.0, value=0.5, step=0.1)
    normalize = st.checkbox(t("normalize"), value=True)
    fps = st.slider(t("fps"), min_value=1, max_value=20, value=6)
    frame_step = st.selectbox(t("frame_granularity"), options=[t("monthly"), t("quarterly"), t("yearly")], index=0)

@st.cache_data
def load_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[['Close']].rename(columns={'Close': 'close'})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def make_savings_curve(start, end, monthly_contrib, annual_interest_pct):
    month_ends = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq='ME')
    r = (annual_interest_pct / 100.0) / 12.0
    balances = []
    balance = 0.0
    for _ in range(len(month_ends)):
        balance = balance * (1 + r) + monthly_contrib
        balances.append(balance)
    savings_monthly = pd.Series(balances, index=month_ends, name='savings')
    bdays = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq='B')
    savings_daily = savings_monthly.reindex(bdays).ffill()
    return savings_daily

prices = load_prices(ticker, start_date, end_date)
if prices.empty:
    st.error(t("no_price_data"))
    st.stop()

savings = make_savings_curve(prices.index.min().date(), prices.index.max().date(), monthly_contrib, annual_interest)
savings = savings.reindex(prices.index).ffill()

df = pd.DataFrame({'price': prices['close'].squeeze(), 'savings': savings})
df_norm = df.copy()
df_norm['price'] = df_norm['price'] / df_norm['price'].iloc[0] * 100.0
df_norm['savings'] = (df_norm['savings'] / max(df_norm['savings'].iloc[0], 1e-9)) * 100.0
plot_df = df_norm if normalize else df

if frame_step == t("monthly"):
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='ME')
elif frame_step == t("quarterly"):
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='QE')
else:
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='YE')

frame_index = [d for d in frame_index if d in plot_df.index]
if not frame_index:
    frame_index = [plot_df.index[-1]]

t0 = frame_index[0]
cut0 = plot_df.loc[:t0]
ytitle = t("indexed_title") if normalize else t("price_savings_title")

fig = go.Figure(
    data=[
        go.Scatter(x=cut0.index, y=cut0['price'], mode='lines', name=f"{ticker}"),
        go.Scatter(x=cut0.index, y=cut0['savings'], mode='lines', name=t("savings_label"))
    ],
    layout=go.Layout(
        title=t("replay_title", ticker=ticker, years=years),
        xaxis=dict(title=t("date")),
        yaxis=dict(title=ytitle),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label=t("play"), method="animate",
                         args=[None, {"frame": {"duration": int(1000/max(fps,1))}, "fromcurrent": True}]),
                    dict(label=t("pause"), method="animate",
                         args=[[None], {"frame": {"duration": 0}}])
                ],
                direction="left",
                showactive=False,
                x=0.0, y=1.2
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": t("up_to")},
                pad={"t": 50},
                steps=[
                    dict(
                        label=dt.strftime("%Y-%m-%d"),
                        method="animate",
                        args=[[dt.strftime("%Y-%m-%d")],
                              {"frame":{"duration": int(1000/max(fps,1)), "redraw": True},
                               "mode":"immediate"}]
                    )
                    for dt in frame_index
                ]
            )
        ]
    ),
    frames=[
        go.Frame(
            name=dt.strftime("%Y-%m-%d"),
            data=[
                go.Scatter(x=plot_df.loc[:dt].index, y=plot_df.loc[:dt, 'price'], name=f"{ticker}"),
                go.Scatter(x=plot_df.loc[:dt].index, y=plot_df.loc[:dt, 'savings'], name=t("savings_label"))
            ]
        )
        for dt in frame_index
    ]
)

st.plotly_chart(fig, use_container_width=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(t("price_change"), f"{(df['price'].iloc[-1]/df['price'].iloc[0]-1)*100:.2f}%")
with col2:
    total_months = len(pd.date_range(start=df.index.min(), end=df.index.max(), freq='ME'))
    st.metric(t("total_contrib"), f"{(monthly_contrib * total_months):,.0f}")
with col3:
    st.metric(t("savings_balance"), f"{df['savings'].iloc[-1]:,.0f}")
