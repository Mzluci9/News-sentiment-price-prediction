from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def publication_frequency(df: pd.DataFrame) -> pd.Series:
    """Return daily publication counts indexed by date."""
    if "date" not in df.columns and "Date" not in df.columns:
        raise KeyError("DataFrame must contain a 'date' or 'Date' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df.get("date", df.get("Date")), errors="coerce")
    df = df.dropna(subset=["date"])
    return df.groupby(df["date"].dt.date).size()


def plot_publication_frequency(daily_counts: pd.Series, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind="line", color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.title("Article Publication Frequency Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_dir / "daily_publication_frequency.png")
    plt.close()


def weekly_publication(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["date"] = pd.to_datetime(df.get("date", df.get("Date")), errors="coerce")
    df = df.dropna(subset=["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    weekly_counts = df["day_of_week"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    return weekly_counts.fillna(0)


def plot_weekly_frequency(weekly_counts: pd.Series, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekly_counts.index, y=weekly_counts.values, palette="viridis")
    plt.xlabel("Day of the Week")
    plt.ylabel("Number of Articles")
    plt.title("Publication Frequency by Day of the Week")
    plt.tight_layout()
    plt.savefig(output_dir / "weekly_publication_frequency.png")
    plt.close()
