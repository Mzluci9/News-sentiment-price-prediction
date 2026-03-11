import pandas as pd
from pathlib import Path

from news_sentiment_price_prediction import data_loader


def test_load_news_data(tmp_path):
    sample = tmp_path / "news.csv"
    sample.write_text("headline,date,publisher\nTest,2020-01-01,Pub")
    df = data_loader.load_news_data(sample)
    assert not df.empty
    assert "headline" in df.columns


def test_load_stock_data(tmp_path):
    sample = tmp_path / "stock.csv"
    sample.write_text("Date,Close\n2020-01-01,100")
    df = data_loader.load_stock_data(sample, ticker="XYZ")
    assert not df.empty
    assert "Ticker" in df.columns
    assert df.loc[0, "Ticker"] == "XYZ"
