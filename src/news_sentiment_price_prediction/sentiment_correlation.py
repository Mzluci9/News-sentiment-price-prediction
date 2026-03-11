from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


def compute_sentiment(df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
    """Add a compound sentiment score column using VADER.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing headlines or text.
    text_column : str
        Name of the column containing text to analyze.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with a new ``Sentiment`` column.
    """
    analyzer = SentimentIntensityAnalyzer()
    new_df = df.copy()
    new_df["Sentiment"] = new_df[text_column].astype(str).apply(
        lambda txt: analyzer.polarity_scores(txt)["compound"]
    )
    return new_df


def align_news_stock(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    ticker: str,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Merge news and stock dataframes on date and ticker, computing returns."""
    nd = news_df.copy()
    sd = stock_df.copy()
    nd[date_column] = pd.to_datetime(nd[date_column], errors="coerce")
    sd[date_column] = pd.to_datetime(sd[date_column], errors="coerce")
    nd = nd.dropna(subset=[date_column])
    sd = sd.dropna(subset=[date_column])
    # average sentiment per day
    news_ticker = (
        nd[nd["Ticker"] == ticker][[date_column, "Sentiment"]]
        .groupby(date_column)
        .mean()
        .reset_index()
    )
    stock_ticker = sd[sd["Ticker"] == ticker][[date_column, "Close"]]
    merged = pd.merge(news_ticker, stock_ticker, on=date_column, how="inner")
    merged["Returns"] = merged["Close"].pct_change()
    return merged


def correlation_plot(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Union[str, Path],
) -> float | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("Empty dataframe passed to correlation_plot")
        return None

    corr = df["Sentiment"].corr(df["Returns"])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Sentiment", y="Returns", data=df, color="blue", alpha=0.5)
    plt.title(f"{ticker} Sentiment vs. Returns (Correlation: {corr:.4f})")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Daily Returns")
    plt.grid(True)
    plt.savefig(output_dir / f"{ticker}_sentiment_returns.png")
    plt.close()
    logger.info("Saved correlation plot for %s", ticker)
    return corr
