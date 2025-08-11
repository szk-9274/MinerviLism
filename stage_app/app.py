"""Streamlit application for visualizing Minervini stages.

This module builds a small dashboard that downloads OHLC data, computes
Minervini's moving–average based stages and renders them as a candlestick
chart.  The UI is intentionally lightweight so that changing the ticker or
lookback period immediately refreshes the visualisation without the need for a
"Run" button.
"""

import warnings
from datetime import datetime
from time import perf_counter, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# パッケージ/単体スクリプト両対応
try:
    from stage_app.stage import (
        STAGE_COLORS,
        classify_stages,
        compute_indicators,
        fetch_price_data,
    )
except ModuleNotFoundError:
    from stage import (
        STAGE_COLORS,
        classify_stages,
        compute_indicators,
        fetch_price_data,
    )

st.set_page_config(layout="wide")

# Custom CSS tweaks for a slightly more modern look-and-feel.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      header[data-testid="stHeader"] { backdrop-filter: blur(8px); }
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

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

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                customdata=df["Stage"],
                hovertext=[
                    f"Open {o:.2f}<br>High {h:.2f}<br>Low {l:.2f}<br>Close {c:.2f}<br>Stage {int(s) if pd.notna(s) else 'NA'}"
                    for o, h, l, c, s in zip(
                        df["Open"], df["High"], df["Low"], df["Close"], df["Stage"]
                    )
                ],
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

    monthly = df["Stage"].resample("MS").apply(
        lambda s: s.mode().iloc[0] if not s.dropna().empty else np.nan
    )
    month_starts = monthly.index.to_list()
    for i, ms in enumerate(month_starts):
        stg = monthly.iloc[i]
        if pd.isna(stg):
            continue
        x0 = ms
        x1 = month_starts[i + 1] if i + 1 < len(month_starts) else df.index[-1]
        color = STAGE_COLORS.get(int(stg), "white")
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=color, opacity=0.10, line_width=0, layer="below"
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # X軸は受け取ったデータ範囲をそのまま表示する（NaNによる短縮を防止）
    fig.update_xaxes(range=[df.index.min(), df.index.max()])
    return fig


def main() -> None:
    """Render the Stage analysis dashboard."""

    # Debounce support -----------------------------------------------------
    if "_changed_at" not in st.session_state:
        st.session_state._changed_at = 0.0

    def _on_change() -> None:
        """Record the timestamp of the latest widget interaction."""
        st.session_state._changed_at = time()

    # ===== サイドバー =====
    st.sidebar.header("Settings")
    label = st.sidebar.selectbox(
        "Ticker", list(CHOICES.keys()), index=0, key="ticker", on_change=_on_change
    )
    ticker = CHOICES[label]
    years = st.sidebar.number_input(
        "期間（年数）", 1, 10, 1, 1, key="years", on_change=_on_change
    )
    slope_smooth_window = st.sidebar.number_input(
        "Slope smoothing window", 1, 50, 5, 1, key="slope", on_change=_on_change
    )

    st.sidebar.markdown("### Stage Colors")
    for s, c in STAGE_COLORS.items():
        st.sidebar.markdown(
            f"<span style='background-color:{c};padding:2px 8px;border-radius:3px;color:white;'>Stage {s}</span>",
            unsafe_allow_html=True,
        )

    # 軽いデバウンス: 直近 0.25 秒以内の多重トリガーは無視
    if time() - st.session_state._changed_at < 0.25:
        st.stop()

    plot_area = st.empty()
    table_area = st.empty()
    start = perf_counter()

    try:
        with plot_area.container():
            st.caption("描画中です…")
            with st.spinner("Loading..."):
                # 1) データ取得（表示よりも長めに取得して指標計算に利用）
                display_days = int(years * 365)
                fetch_days = display_days + 400
                data = cached_fetch(ticker, lookback_days=fetch_days)

                # 2) 指標計算（Closeベース）
                df = compute_indicators(data)

                # 3) ステージ分類
                df["Stage"] = classify_stages(
                    df, slope_smooth_window=int(slope_smooth_window)
                )

                # 4) 描画
                need = ["Open", "High", "Low", "Close"]
                df_plot = df.dropna(subset=need).copy()
                if df_plot.empty:
                    st.info(
                        "まだステージが計算できた行がありません（SMAや200日傾きの計算に十分な日数が必要です）。"
                    )
                    return

                display_end = df_plot.index.max()
                # Trim only after indicators and stages are computed so the
                # warm-up period fetched in ``fetch_price_data`` is preserved.
                display_start = display_end - pd.Timedelta(days=display_days)
                df_plot = df_plot[
                    (df_plot.index >= display_start) & (df_plot.index <= display_end)
                ]
                fig = build_chart(df_plot, ticker)
                fig.update_layout(template="plotly_dark")

        elapsed = perf_counter() - start

        # ===== ヘッダー行 =====
        header = st.container()
        with header:
            left, right = st.columns([0.7, 0.3])
            with left:
                st.markdown(f"## {ticker}")
                st.markdown(
                    f"<span style='background-color:#444;padding:2px 6px;border-radius:4px;'> {int(years)}Y </span>",
                    unsafe_allow_html=True,
                )
            with right:
                st.caption(f"{datetime.now():%Y-%m-%d %H:%M:%S} • {elapsed:.2f}s")

        # ===== グラフ描画 =====
        with plot_area:
            st.plotly_chart(fig, use_container_width=True)

        # ===== 凡例 =====
        legend = st.container()
        with legend:
            cols = st.columns(len(STAGE_COLORS) + 3)
            for i, (stage, color) in enumerate(STAGE_COLORS.items()):
                cols[i].markdown(
                    f"<div style='display:flex;align-items:center;'>"
                    f"<span style='width:12px;height:12px;background:{color};"
                    f"display:inline-block;margin-right:4px;'></span>Stage {stage}</div>",
                    unsafe_allow_html=True,
                )
            cols[len(STAGE_COLORS)].markdown("MA25")
            cols[len(STAGE_COLORS) + 1].markdown("MA50")
            cols[len(STAGE_COLORS) + 2].markdown("MA200")

        # ===== テーブル =====
        tbl = (
            df_plot[["Close", "SMA25", "SMA50", "SMA200", "Stage"]]
            .dropna(subset=["SMA25", "SMA50", "SMA200", "Stage"])
            .tail(10)
        )
        with table_area:
            st.dataframe(tbl)

        st.toast(f"Updated {ticker} ({int(years)}Y)")

    except Exception as exc:  # 取得失敗や列欠損などはここで通知
        with plot_area.container():
            st.error("Failed to update chart.")
            with st.expander("詳細を開く"):
                st.write(str(exc))


if __name__ == "__main__":
    main()
