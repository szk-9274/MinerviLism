"""Command line interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .indicators import compute_indicators
from .io_utils import LoadConfig, load_data
from .plotter import plot_stages, plot_vcp
from .stage_rules import StageConfig, classify_stage
from .vcp import VCPConfig, detect_vcp

app = typer.Typer()


def _load(ticker: Optional[str], years: int, csv: Optional[Path]) -> pd.DataFrame:
    cfg = LoadConfig(ticker=ticker, years=years, csv_path=str(csv) if csv else None)
    return load_data(cfg)


@app.command()
def classify(ticker: Optional[str] = None, years: int = 5, csv: Optional[Path] = None, out: Path = Path("out.csv")) -> None:
    df = _load(ticker, years, csv)
    df = compute_indicators(df)
    df = classify_stage(df, StageConfig())
    df.to_csv(out)
    typer.echo(f"Saved {out}")


@app.command()
def vcp(ticker: Optional[str] = None, years: int = 5, csv: Optional[Path] = None, json_out: Path = Path("vcp.json")) -> None:
    df = _load(ticker, years, csv)
    df = compute_indicators(df)
    df_vcp, bases = detect_vcp(df, VCPConfig())
    bases.to_json(json_out, orient="records", date_format="iso")
    typer.echo(f"Saved {json_out}")


@app.command()
def plot(ticker: Optional[str] = None, years: int = 5, csv: Optional[Path] = None, out: Path = Path("chart.png"), with_vcp: bool = False) -> None:
    import matplotlib.pyplot as plt

    df = _load(ticker, years, csv)
    df = compute_indicators(df)
    df = classify_stage(df, StageConfig())
    if with_vcp:
        df, bases = detect_vcp(df, VCPConfig())
        ax = plot_vcp(df, bases)
    else:
        ax = plot_stages(df)
    ax.figure.savefig(out)
    typer.echo(f"Saved {out}")


if __name__ == "__main__":
    app()
