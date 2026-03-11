from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pandas_ta is small dependency alternative
    ta = None

logger = logging.getLogger(__name__)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to a stock price DataFrame.

    The returned frame will contain columns for simple moving averages
    (20, 50), exponential moving average (20), RSI (14), MACD and related
    signals.  A few naive buy/sell signals based on crossovers are also
    inserted for demonstration.

    The function requires ``pandas_ta`` to be installed; otherwise it will
    raise ``RuntimeError``.
    """
    if ta is None:
        raise RuntimeError("pandas_ta must be installed to compute indicators")

    df = df.copy()
    # ensure required numeric columns exist
    for col in ["Close"]:
        if col not in df.columns:
            logger.warning("Column %s missing from DataFrame", col)
            df[col] = np.nan

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    df["MACD"] = macd.get("MACD_12_26_9")
    df["MACD_Signal"] = macd.get("MACDs_12_26_9")
    df["MACD_Hist"] = macd.get("MACDh_12_26_9")

    # example signals
    df["SMA_Crossover"] = np.where(df["SMA_20"] > df["SMA_50"], 1, -1)
    df["RSI_Signal"] = np.select([
        df["RSI_14"] > 70,
        df["RSI_14"] < 30,
    ],
    [-1, 1], default=0)
    logger.info("Technical indicators computed")
    return df


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for core indicator columns."""
    cols = [c for c in ["Close", "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD"] if c in df.columns]
    return df[cols].describe()


def save_plots(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Generate and save a set of standard plots from indicator data."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df.get("Close"), label="Close Price", color="blue")
    plt.plot(df["Date"], df.get("SMA_20"), label="SMA 20", color="orange")
    plt.plot(df["Date"], df.get("SMA_50"), label="SMA 50", color="green")
    plt.plot(df["Date"], df.get("EMA_20"), label="EMA 20", color="red")
    plt.title("Close Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "price_ma.png")
    plt.close()

    # RSI
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df.get("RSI_14"), label="RSI 14", color="purple")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.title("RSI Indicator")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "rsi.png")
    plt.close()

    # MACD
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df.get("MACD"), label="MACD", color="blue")
    plt.plot(df["Date"], df.get("MACD_Signal"), label="Signal", color="orange")
    plt.bar(df["Date"], df.get("MACD_Hist"), label="Histogram", color="grey", alpha=0.3)
    plt.title("MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "macd.png")
    plt.close()
