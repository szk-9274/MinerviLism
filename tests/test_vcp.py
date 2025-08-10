import numpy as np
import pandas as pd

from minervini_stage.vcp import detect_vcp, VCPConfig

def test_vcp_detection():
    dates = pd.date_range('2020-01-01', periods=61)
    close = np.interp(range(61), [0,10,20,30,40,45,60], [100,90,97,93,96,94,100])
    volume = np.array([1000]*30 + [500]*31)
    df = pd.DataFrame({'Close': close, 'Volume': volume}, index=dates)
    df_vcp, bases = detect_vcp(df, VCPConfig(min_base_len=30, max_base_len=55, lookback=3))
    assert not bases.empty
    assert df_vcp['is_VCP_pivot_breakout'].any()
