import warnings
warnings.filterwarnings("ignore")

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stage_app.stage import (
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

    segments = []
    start = df.index[0]
    current = df["Stage"].iloc[0]
    for idx, stage in df["Stage"].iloc[1:].items():
        if stage != current:
            segments.append((start, idx, current))
            start = idx
            current = stage
    segments.append((start, df.index[-1], current))

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
            fig = build_chart(df.dropna())
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                df[["Close", "SMA50", "SMA150", "SMA200", "Stage"]].dropna().tail(10)
            )
        pbar.progress(100)

        total = perf_counter() - start_time
        st.write(f"Completed in {total:.2f}s")
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))


if __name__ == "__main__":
    main()
