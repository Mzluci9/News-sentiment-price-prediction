import pandas as pd
from pathlib import Path
import sys, os
# make sure the package in src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from news_sentiment_price_prediction import (
    descriptive,
    text_analysis,
    time_series,
    publisher,
    technical_indicators,
    sentiment_correlation,
)


def create_sample_news():
    return pd.DataFrame(
        {
            "headline": ["Earnings growth", "FDA approves", "Nothing"],
            "date": ["2021-01-01", "2021-01-02", "2021-01-02"],
            "publisher": ["A", "B", "C"],
        }
    )


def create_sample_stock():
    return pd.DataFrame(
        {
            "Date": ["2021-01-01", "2021-01-02"],
            "Close": [100, 105],
            "Ticker": ["AAPL", "AAPL"],
        }
    )


def test_headline_length_statistics():
    df = create_sample_news()
    stats = descriptive.headline_length_statistics(df)
    assert "mean" in stats


def test_clean_headlines_and_lda():
    df = create_sample_news()
    cleaned = text_analysis.clean_headlines(df["headline"])
    assert isinstance(cleaned, pd.Series)
    topics = text_analysis.lda_topics(cleaned)
    assert isinstance(topics, list)


def test_publication_frequency():
    df = create_sample_news()
    freq = time_series.publication_frequency(df)
    assert freq.iloc[0] == 1


def test_top_publishers():
    df = create_sample_news()
    counts = publisher.top_publishers(df, top_n=2)
    assert list(counts.index) == ["A", "B"]


def test_technical_indicators(monkeypatch):
    # patch pandas_ta to avoid requiring installation
    class Dummy:
        @staticmethod
        def sma(series, length):
            return series * 0

        @staticmethod
        def ema(series, length):
            return series * 0

        @staticmethod
        def rsi(series, length):
            return series * 0

        @staticmethod
        def macd(series):
            return {"MACD_12_26_9": series * 0, "MACDs_12_26_9": series * 0, "MACDh_12_26_9": series * 0}

    monkeypatch.setattr(technical_indicators, "ta", Dummy)
    df = create_sample_stock()
    df_ind = technical_indicators.compute_technical_indicators(df)
    assert "SMA_20" in df_ind.columns


def test_sentiment_correlation():
    news = create_sample_news()
    news = sentiment_correlation.compute_sentiment(news, text_column="headline")
    stock = create_sample_stock()
    merged = sentiment_correlation.align_news_stock(news, stock, "AAPL", date_column="date")
    assert "Sentiment" in merged.columns

