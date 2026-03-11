# News-Sentiment-Price-Prediction

This repository contains tools for analyzing news headlines and stock price
data with the goal of investigating correlations between sentiment and
market movements. The original analysis was performed interactively in
Jupyter notebooks; the code has been refactored into a small Python
package and accompanying command-line script.

## Structure

```
README.md
requirements.txt
src/news_sentiment_price_prediction/    # package code
notebooks/                              # example notebooks (legacy)
data/                                   # CSV datasets
tests/                                  # unit tests
```

## Installation

1. Create and activate a virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

A simple CLI is provided for running the analyses on your own data:

```bash
python -m news_sentiment_price_prediction.main \
    --news data/raw_analyst_ratings.csv \
    --stock data/processed_AAPL_data.csv \
    --output plots/
```

The script will generate descriptive statistics, word clouds, frequency
plots, technical indicator plots, and sentiment/return correlation graphs
in the specified output directory.

You can also import individual functions from
`news_sentiment_price_prediction` in your own code or use the
notebooks as examples.

## Testing

Run the unit tests with pytest:

```bash
pytest
```

## Notes

* Absolute paths have been removed in favor of parameterized file paths.
* The notebooks remain for reference but are no longer the primary
  execution mechanism.
* Dependencies are listed in `requirements.txt`; they should be updated
  periodically to stay current with security patches and new features.

