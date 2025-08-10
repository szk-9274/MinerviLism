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


@st.cache_data(ttl=3600)
def cached_fetch(ticker: str) -> pd.DataFrame:
    return fetch_price_data(ticker)


def build_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        raise ValueError("No rows with computed Stage yet. Need enough data to compute indicators.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
            line=dict(color="blue"),
            customdata=df["Stage"],
            hovertemplate="Price: %{y:.2f}<br>Stage: %{customdata}<extra></extra>",
        )
    )

    # 区間塗り分け（データ1点でも落ちない）
    segments = []
    idxs = df.index
    stages = df["Stage"]

    start = idxs[0]
    current = stages.iloc[0]
    for i in range(1, len(df)):
        if stages.iloc[i] != current:
            segments.append((start, idxs[i], current))
            start = idxs[i]
            current = stages.iloc[i]
    segments.append((start, idxs[-1], current))

    for s, e, stage in segments:
        color = STAGE_COLORS.get(stage, "white")
        fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=0.1, line_width=0)

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=40))
    return fig



def main() -> None:
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    run_btn = st.sidebar.button("Run")

    st.sidebar.markdown("### Stage Colors")
    for s, c in STAGE_COLORS.items():
        st.sidebar.markdown(
            f"<span style='background-color:{c};padding:2px 8px;border-radius:3px;color:white;'>Stage {s}</span>",
            unsafe_allow_html=True,
        )

    if not run_btn:
        return

    pbar = st.progress(0)
    steps = 5
    start_time = perf_counter()

    # 1. Validate input
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        return
    pbar.progress(int(100 / steps))

    try:
        # 2. Downloading data
        with st.spinner("Downloading data..."):
            data = cached_fetch(ticker)
        pbar.progress(int(2 * 100 / steps))

        # 3. Computing indicators
        with st.spinner("Computing indicators..."):
            df = compute_indicators(data)
        pbar.progress(int(3 * 100 / steps))

        # 4. Classifying stages
        with st.spinner("Classifying stages..."):
            df["Stage"] = classify_stages(df)
        pbar.progress(int(4 * 100 / steps))

        # 5. Rendering chart
        with st.spinner("Rendering chart..."):
            # グラフ用は Close / Stage だけで欠損を落とす
            df_plot = df[["Close", "Stage"]].dropna(subset=["Close", "Stage"])

            if df_plot.empty:
                st.info("まだステージが計算できた行がありません（SMAや200日傾きの計算に十分な日数が必要です）。")
            else:
                fig = build_chart(df_plot)
                st.plotly_chart(fig, use_container_width=True)

            # テーブルは移動平均とStageが出た行だけ
            tbl = (
                df[["Close", "SMA50", "SMA150", "SMA200", "Stage"]]
                .dropna(subset=["SMA50", "SMA150", "SMA200", "Stage"])
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
