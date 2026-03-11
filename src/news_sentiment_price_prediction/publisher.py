from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def top_publishers(df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    return df.get("publisher", pd.Series(dtype="object")).value_counts().head(top_n)


def plot_top_publishers(counts: pd.Series, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, color="tab:green")
    plt.xlabel("Number of Articles")
    plt.ylabel("Publisher")
    plt.title("Top Publishers by Article Count")
    plt.tight_layout()
    plt.savefig(output_dir / "top_publishers.png")
    plt.close()


def extract_email_domains(df: pd.DataFrame) -> pd.Series:
    """If any publisher values look like email addresses, return top domains."""
    series = df.get("publisher", pd.Series(dtype="object"))
    has_email = series.str.contains(r"@", na=False)
    if not has_email.any():
        return pd.Series(dtype="int")
    domains = series[has_email].str.split("@").str[1]
    return domains.value_counts()


def categorize_news_types(df: pd.DataFrame) -> pd.Series:
    """Label headlines with simple categories and return counts."""
    def categorize(headline: str) -> str:
        text = (headline or "").lower()
        if "earning" in text:
            return "Earnings"
        if "price target" in text:
            return "Price Target"
        if "fda" in text:
            return "FDA"
        return "Other"

    types = df.get("headline", pd.Series(dtype="object")).astype(str).apply(categorize)
    df = df.copy()
    df["news_type"] = types
    return df["news_type"].value_counts()


def plot_news_types_by_publisher(df: pd.DataFrame, output_dir: Union[str, Path], top_n: int = 5) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = top_publishers(df, top_n)
    top = counts.index
    subset = df[df["publisher"].isin(top)]
    grouped = subset.groupby(["publisher", "news_type"]).size().unstack(fill_value=0)
    grouped.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.xlabel("Publisher")
    plt.ylabel("Number of Articles")
    plt.title("News Types Reported by Top Publishers")
    plt.legend(title="News Type")
    plt.tight_layout()
    plt.savefig(output_dir / "news_types_by_publisher.png")
    plt.close()
