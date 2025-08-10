import pandas as pd
from minervini_stage.stage_rules import classify_stage, StageConfig

def make_row(**kwargs):
    return pd.DataFrame([kwargs])

def test_stage_classification():
    cfg = StageConfig()
    df2 = make_row(Close=100,SMA50=90,SMA150=80,SMA200=70,SMA200_slope=1,SMA200_slope_norm=0.01,High52w=120,Low52w=60)
    assert classify_stage(df2,cfg)['Stage'].iloc[0]==2
    df4 = make_row(Close=60,SMA50=80,SMA150=85,SMA200=70,SMA200_slope=-1,SMA200_slope_norm=-0.01,High52w=100,Low52w=50)
    assert classify_stage(df4,cfg)['Stage'].iloc[0]==4
    df3 = make_row(Close=75,SMA50=80,SMA150=82,SMA200=70,SMA200_slope=0,SMA200_slope_norm=0,High52w=90,Low52w=60)
    assert classify_stage(df3,cfg)['Stage'].iloc[0]==3
    df1 = make_row(Close=65,SMA50=70,SMA150=75,SMA200=80,SMA200_slope=0.5,SMA200_slope_norm=0.01,High52w=90,Low52w=60)
    assert classify_stage(df1,cfg)['Stage'].iloc[0]==1
