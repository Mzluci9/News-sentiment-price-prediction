"""Simple command-line interface for running analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data_loader, descriptive, publisher, sentiment_correlation, technical_indicators, text_analysis, time_series, utils


def main():
    parser = argparse.ArgumentParser(description="Run analyses on news and stock data")
    parser.add_argument("--news", type=Path, help="Path to news CSV file", required=True)
    parser.add_argument("--stock", type=Path, help="Path to stock CSV file", required=False)
    parser.add_argument("--output", type=Path, help="Directory for output plots", default=Path("plots"))
    args = parser.parse_args()

    news_path = args.news
    stock_path = args.stock
    out_dir = utils.ensure_directory(args.output)

    news_df = data_loader.load_news_data(news_path)
    if news_df.empty:
        print("No news data loaded, aborting")
        return

    # Descriptive statistics
    print("Headline length stats:\n", descriptive.headline_length_statistics(news_df))
    descriptive.plot_headline_length(news_df, out_dir)
    print("Top publishers:\n", descriptive.publisher_counts(news_df))

    # Text analysis wordcloud
    import pandas as pd  # local import to avoid heavyweight dependency at top-level

    cleaned = text_analysis.clean_headlines(news_df.get("headline", pd.Series(dtype="object")))
    try:
        text_analysis.create_wordcloud(" ".join(cleaned), out_dir / "wordcloud.png")
    except ValueError:
        print("No valid text for word cloud")

    # Time-series and publisher
    daily_counts = time_series.publication_frequency(news_df)
    time_series.plot_publication_frequency(daily_counts, out_dir)
    weekly_counts = time_series.weekly_publication(news_df)
    time_series.plot_weekly_frequency(weekly_counts, out_dir)

    if stock_path:
        stock_df = data_loader.load_stock_data(stock_path)
        tickers = stock_df["Ticker"].unique() if not stock_df.empty else []
        for tick in tickers:
            df_ind = technical_indicators.compute_technical_indicators(stock_df[stock_df["Ticker"] == tick])
            technical_indicators.save_plots(df_ind, out_dir)

            # Sentiment correlation if sentiment available
            news_sent = sentiment_correlation.compute_sentiment(news_df)
            merged = sentiment_correlation.align_news_stock(news_sent, stock_df, tick)
            sentiment_correlation.correlation_plot(merged, tick, out_dir)

    print("Analysis complete. See plots in", out_dir)


if __name__ == "__main__":
    main()
