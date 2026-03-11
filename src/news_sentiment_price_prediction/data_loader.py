from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_news_data(path: Path) -> pd.DataFrame:
    """Load news dataset from a CSV file.

    Parameters
    ----------
    path : Path
        Path to the CSV file containing news (raw analyst ratings).

    Returns
    -------
    pd.DataFrame
        Dataframe of news; returns empty dataframe on failure.
    """
    try:
        df = pd.read_csv(path)
        logger.info("Loaded news data from %s", path)
        return df
    except FileNotFoundError:
        logger.error("News file not found: %s", path)
    except pd.errors.EmptyDataError:
        logger.error("News file is empty: %s", path)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error loading news data: %s", exc)
    return pd.DataFrame()


def load_stock_data(path: Path, ticker: str = "AAPL") -> pd.DataFrame:
    """Load historical stock data and normalize columns.

    Adds a constant ``Ticker`` column if missing and ensures ``Date`` is datetime.

    Parameters
    ----------
    path : Path
        Path to stock CSV file.
    ticker : str, optional
        Ticker symbol to assign to rows without ticker information.

    Returns
    -------
    pd.DataFrame
        Stock price dataframe; empty if load fails.
    """
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Ticker" not in df.columns:
            df["Ticker"] = ticker
        df.sort_values(by="Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Loaded stock data from %s", path)
        return df
    except FileNotFoundError:
        logger.error("Stock file not found: %s", path)
    except pd.errors.EmptyDataError:
        logger.error("Stock file is empty: %s", path)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error loading stock data: %s", exc)
    return pd.DataFrame()
