import warnings
warnings.filterwarnings("ignore")

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:  # Attempt absolute import when package is installed
    from stage_app.stage import (
        STAGE_COLORS,
        classify_stages,
        compute_indicators,
        fetch_price_data,
    )
except ModuleNotFoundError:  # Fallback for running as a script
    from stage import (
        STAGE_COLORS,
        classify_stages,
        compute_indicators,
        fetch_price_data,
    )

st.set_page_config(layout="wide")


CHOICES = {
    "NVIDIA (NVDA)": "NVDA",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Bitcoin (BTC-USD)": "BTC-USD",
}


@st.cache_data(ttl=1800)
def cached_fetch(ticker: str, lookback_days: int) -> pd.DataFrame:
    return fetch_price_data(ticker, lookback_days=lookback_days)


def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    if df.empty:
        raise ValueError("No rows with computed Stage yet. Need enough data to compute indicators.")

    hovertext = (
        "Open "
        + df["Open"].round(2).astype(str)
        + "<br>High "
        + df["High"].round(2).astype(str)
        + "<br>Low "
        + df["Low"].round(2).astype(str)
        + "<br>Close "
        + df["Close"].round(2).astype(str)
        + "<br>Stage "
        + df["Stage"].astype(int).astype(str)
    )

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                text=hovertext,
                hoverinfo="text",
            )
        ]
    )

    for name, color in [
        ("SMA25", "blue"),
        ("SMA50", "orange"),
        ("SMA200", "purple"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[name],
                name=name,
                mode="lines",
                line=dict(color=color),
            )
        )

    # 月ごとの区間塗り分け。月内の最頻値ステージを代表値として採用。
    groups = df.groupby(df.index.to_period("M"))
    bounds = [(g.index[0], g.index[-1]) for _, g in groups]
    stage_mode = groups["Stage"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).values
    for (start, end), stage in zip(bounds, stage_mode):
        color = STAGE_COLORS.get(stage, "white")
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.1,
            line_width=0,
            layer="below",
        )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=40), xaxis_rangeslider_visible=False)
    return fig



def main() -> None:
    st.sidebar.header("Settings")
    label = st.sidebar.selectbox("Ticker", list(CHOICES.keys()), index=0)
    ticker = CHOICES[label]
    years = st.sidebar.number_input("期間（年数）", 1, 10, 1, 1)
    run_btn = st.sidebar.button("Run")
    st.sidebar.markdown("### Stage Colors")
    for s, c in STAGE_COLORS.items():
        st.sidebar.markdown(
            f"<span style='background-color:{c};padding:2px 8px;border-radius:3px;color:white;'>Stage {s}</span>",
            unsafe_allow_html=True,
        )

    pbar = st.progress(0)
    steps = 4
    start_time = perf_counter()

    try:
        if run_btn or "df" not in st.session_state:
            lookback_days = int(years * 365)
            with st.spinner("Downloading data..."):
                data = cached_fetch(ticker, lookback_days)
            pbar.progress(int(1 * 100 / steps))

            with st.spinner("Computing indicators..."):
                df = compute_indicators(data)
            pbar.progress(int(2 * 100 / steps))

            with st.spinner("Classifying stages..."):
                df["Stage"] = classify_stages(df)
            pbar.progress(int(3 * 100 / steps))

            df = df.dropna(subset=["Open", "High", "Low", "Close", "Stage"])
            st.session_state["df"] = df

        df = st.session_state.get("df", pd.DataFrame())

        with st.spinner("Rendering chart..."):
            if df.empty:
                st.info("まだステージが計算できた行がありません（SMAや200日傾きの計算に十分な日数が必要です）。")
            else:
                fig = build_chart(df, ticker)
                st.plotly_chart(fig, use_container_width=True)

            tbl = (
                df[["Close", "SMA25", "SMA50", "SMA200", "Stage"]]
                .dropna(subset=["SMA25", "SMA50", "SMA200", "Stage"])
                .tail(10)
            )
            st.dataframe(tbl)

        pbar.progress(100)

        total = perf_counter() - start_time
        st.write(f"Completed in {total:.2f}s")
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))


if __name__ == "__main__":
    main()
