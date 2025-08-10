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


def build_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        raise ValueError("No rows with computed Stage yet. Need enough data to compute indicators.")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="NVDA",
                customdata=df["Stage"],
                hovertemplate="Open %{open:.2f}<br>High %{high:.2f}<br>Low %{low:.2f}<br>Close %{close:.2f}<br>Stage %{customdata}<extra></extra>",
            )
        ]
    )

    # 月ごとの区間塗り分け。月末のステージで代表させる。
    groups = df.groupby(df.index.to_period("M"))
    bounds = [(g.index[0], g.index[-1]) for _, g in groups]
    month_ends = [b[1] for b in bounds]
    stage_last = df["Stage"].resample("M").last().reindex(month_ends)
    for (start, end), stage in zip(bounds, stage_last):
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
    st.sidebar.markdown("Ticker: NVDA")
    st.sidebar.markdown("### Stage Colors")
    for s, c in STAGE_COLORS.items():
        st.sidebar.markdown(
            f"<span style='background-color:{c};padding:2px 8px;border-radius:3px;color:white;'>Stage {s}</span>",
            unsafe_allow_html=True,
        )
    ticker = "NVDA"

    pbar = st.progress(0)
    steps = 4
    start_time = perf_counter()

    try:
        # 1. Downloading data
        with st.spinner("Downloading data..."):
            data = fetch_price_data(ticker)
        pbar.progress(int(1 * 100 / steps))

        # 2. Computing indicators
        with st.spinner("Computing indicators..."):
            df = compute_indicators(data)
        pbar.progress(int(2 * 100 / steps))

        # 3. Classifying stages
        with st.spinner("Classifying stages..."):
            df["Stage"] = classify_stages(df)
        pbar.progress(int(3 * 100 / steps))

        # 4. Rendering chart
        with st.spinner("Rendering chart..."):
            df_plot = df.dropna(subset=["Stage"]).copy()
            if df_plot.empty:
                st.info("まだステージが計算できた行がありません（SMAや200日傾きの計算に十分な日数が必要です）。")
            else:
                fig = build_chart(df_plot)
                st.plotly_chart(fig, use_container_width=True)

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
