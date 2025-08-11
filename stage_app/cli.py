import warnings
from pathlib import Path

import typer

from stage_app.stage import classify_stages, compute_indicators, fetch_price_data

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Stage classification utilities."""


@app.command()
def classify(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    csv_out: Path = typer.Option(..., help="Output CSV file"),
    slope_smooth_window: int = typer.Option(
        5, "--slope-smooth-window", help="Rolling window to smooth 200MA slope"
    ),
    suppress_warnings: bool = typer.Option(
        False, "--suppress-warnings", help="Silence warnings"
    ),
) -> None:
    """Export 1Y stage classification to CSV."""
    if suppress_warnings:
        warnings.filterwarnings("ignore")
    data = fetch_price_data(ticker)
    df = compute_indicators(data)
    df["Stage"] = classify_stages(df, slope_smooth_window=slope_smooth_window)
    df[["Close", "SMA50", "SMA150", "SMA200", "Stage"]].dropna().to_csv(
        csv_out, index_label="Date"
    )
    typer.echo(f"Saved {csv_out}")


if __name__ == "__main__":
    app()
