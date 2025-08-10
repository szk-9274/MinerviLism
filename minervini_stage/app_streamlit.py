"""Streamlit viewer for Minervini stages and VCP."""
from __future__ import annotations

import streamlit as st

try:  # pragma: no cover - runtime import fix
    from .indicators import compute_indicators
    from .io_utils import LoadConfig, load_data
    from .plotter import plot_stages, plot_vcp
    from .stage_rules import StageConfig, classify_stage
    from .vcp import VCPConfig, detect_vcp
except ImportError:  # executed when run as a script outside the package
    from indicators import compute_indicators
    from io_utils import LoadConfig, load_data
    from plotter import plot_stages, plot_vcp
    from stage_rules import StageConfig, classify_stage
    from vcp import VCPConfig, detect_vcp


def main() -> None:
    st.title("Minervini Stage & VCP Viewer")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    years = st.sidebar.number_input("Years", 1, 20, 10)
    uploaded = st.sidebar.file_uploader("CSV")

    if uploaded is not None:
        cfg = LoadConfig(csv_path=uploaded)
    else:
        cfg = LoadConfig(ticker=ticker, years=years)
    df = load_data(cfg)
    df = compute_indicators(df)
    df = classify_stage(df, StageConfig())
    df_vcp, bases = detect_vcp(df, VCPConfig())

    tab1, tab2 = st.tabs(["Stages", "VCP"])
    with tab1:
        st.pyplot(plot_stages(df_vcp).figure)
    with tab2:
        st.pyplot(plot_vcp(df_vcp, bases).figure)
        st.dataframe(bases)


if __name__ == "__main__":
    main()
