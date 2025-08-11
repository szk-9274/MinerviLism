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
    suppress_warnings: bool = typer.Option(
        False, "--suppress-warnings", help="Silence warnings"
    ),
) -> None:
    """Export 1Y stage classification to CSV."""
    def _run() -> None:
        data = fetch_price_data(ticker)
        df = compute_indicators(data)
        df["Stage"] = classify_stages(df)
        df[["Close", "SMA50", "SMA150", "SMA200", "Stage"]].dropna().to_csv(
            csv_out, index_label="Date"
        )
        typer.echo(f"Saved {csv_out}")

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run()
    else:
        _run()


if __name__ == "__main__":
    app()
