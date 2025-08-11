import warnings
warnings.filterwarnings("ignore")

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    # ===== サイドバー =====
    st.sidebar.header("Settings")
    label = st.sidebar.selectbox("Ticker", list(CHOICES.keys()), index=0)
    ticker = CHOICES[label]
    years = st.sidebar.number_input("期間（年数）", 1, 10, 1, 1)
    slope_smooth_window = st.sidebar.number_input(
        "Slope smoothing window", 1, 50, 5, 1
    )
    run_btn = st.sidebar.button("Run")

    st.sidebar.markdown("### Stage Colors")
    for s, c in STAGE_COLORS.items():
        st.sidebar.markdown(
            f"<span style='background-color:{c};padding:2px 8px;border-radius:3px;color:white;'>Stage {s}</span>",
            unsafe_allow_html=True,
        )

    if not run_btn:
        st.info("左の設定を選んで『Run』を押してください。")
        return

    pbar = st.progress(0)
    steps = 4
    t0 = perf_counter()

    try:
        # 1) データ取得（表示よりも長めに取得して指標計算に利用）
        with st.spinner("Downloading data..."):
            lookback_days = int(years * 365)
            data = cached_fetch(ticker, lookback_days=lookback_days)
        pbar.progress(int(1 * 100 / steps))

        # 2) 指標計算（Closeベース）
        with st.spinner("Computing indicators..."):
            df = compute_indicators(data)
        pbar.progress(int(2 * 100 / steps))

        # 3) ステージ分類
        with st.spinner("Classifying stages..."):
            df["Stage"] = classify_stages(
                df, slope_smooth_window=int(slope_smooth_window)
            )
        pbar.progress(int(3 * 100 / steps))

        # 4) 描画
        with st.spinner("Rendering chart..."):
            # Stage 列の NaN でフィルタすると初期データが消えてしまうため、
            # OHLC が揃っているかのみ確認する
            need = ["Open", "High", "Low", "Close"]
            df_plot = df.dropna(subset=need).copy()
            if df_plot.empty:
                st.info("まだステージが計算できた行がありません（SMAや200日傾きの計算に十分な日数が必要です）。")
            else:
                display_end = df_plot.index.max()
                # Trim only after indicators and stages are computed so the
                # warm-up period fetched in ``fetch_price_data`` is preserved.
                display_start = display_end - pd.Timedelta(days=lookback_days)
                df_plot = df_plot[(df_plot.index >= display_start) & (df_plot.index <= display_end)]
                fig = build_chart(df_plot, ticker)
                st.plotly_chart(fig, use_container_width=True)
                tbl = (
                    df_plot[["Close", "SMA25", "SMA50", "SMA200", "Stage"]]
                    .dropna(subset=["SMA25", "SMA50", "SMA200", "Stage"])
                    .tail(10)
                )
                st.dataframe(tbl)

        pbar.progress(100)
        st.write(f"Completed in {perf_counter() - t0:.2f}s")

    except Exception as exc:  # 取得失敗や列欠損などはここで通知
        st.warning(str(exc))


if __name__ == "__main__":
    main()
