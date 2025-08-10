
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="10-Year Stock Replay & Savings Overlay", layout="wide")
st.title("📈 10-Year Stock Replay & Savings Overlay")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance symbol)", value="SPY")
    years = st.slider("Years of history", min_value=3, max_value=20, value=10, step=1)
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=(end_date - timedelta(days=int(365.25*years))))
    st.markdown("---")
    st.subheader("Savings model")
    monthly_contrib = st.number_input("Monthly contribution", min_value=0.0, value=30000.0, step=1000.0, format="%.0f")
    annual_interest = st.number_input("Annual interest (%)", min_value=0.0, max_value=20.0, value=0.5, step=0.1)
    normalize = st.checkbox("Normalize to 100 at start", value=True)
    fps = st.slider("Playback speed (frames/sec)", min_value=1, max_value=20, value=6)
    frame_step = st.selectbox("Frame granularity", options=["Monthly","Quarterly","Yearly"], index=0)

@st.cache_data
def load_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[['Close']].rename(columns={'Close':'close'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def make_savings_curve(start, end, monthly_contrib, annual_interest_pct):
    month_ends = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq='M')
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
    st.error("No price data found. Check the ticker or date range.")
    st.stop()

savings = make_savings_curve(prices.index.min().date(), prices.index.max().date(), monthly_contrib, annual_interest)
savings = savings.reindex(prices.index).ffill()

df = pd.DataFrame({'price': prices['close'], 'savings': savings})
df_norm = df.copy()
df_norm['price'] = df_norm['price'] / df_norm['price'].iloc[0] * 100.0
df_norm['savings'] = (df_norm['savings'] / max(df_norm['savings'].iloc[0], 1e-9)) * 100.0
plot_df = df_norm if normalize else df

if frame_step == "Monthly":
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='M')
elif frame_step == "Quarterly":
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='Q')
else:
    frame_index = pd.date_range(start=plot_df.index.min(), end=plot_df.index.max(), freq='Y')

frame_index = [d for d in frame_index if d in plot_df.index]
if not frame_index:
    frame_index = [plot_df.index[-1]]

t0 = frame_index[0]
cut0 = plot_df.loc[:t0]
ytitle = "Indexed to 100" if normalize else "Price / Savings"

fig = go.Figure(
    data=[
        go.Scatter(x=cut0.index, y=cut0['price'], mode='lines', name=f"{ticker}"),
        go.Scatter(x=cut0.index, y=cut0['savings'], mode='lines', name="Savings")
    ],
    layout=go.Layout(
        title=f"{ticker} — {years}Y Replay with Savings Overlay",
        xaxis=dict(title="Date"),
        yaxis=dict(title=ytitle),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": int(1000/max(fps,1))}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
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
                currentvalue={"prefix":"Up to: "},
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
                go.Scatter(x=plot_df.loc[:dt].index, y=plot_df.loc[:dt, 'price']),
                go.Scatter(x=plot_df.loc[:dt].index, y=plot_df.loc[:dt, 'savings'])
            ]
        )
        for dt in frame_index
    ]
)

st.plotly_chart(fig, use_container_width=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Price change", f"{(df['price'].iloc[-1]/df['price'].iloc[0]-1)*100:.2f}%")
with col2:
    st.metric("Total contributions", f"{(monthly_contrib * len(pd.date_range(start=df.index.min(), end=df.index.max(), freq='M'))):,.0f}")
with col3:
    st.metric("Savings balance", f"{df['savings'].iloc[-1]:,.0f}")
