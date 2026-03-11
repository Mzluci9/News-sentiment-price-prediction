from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def headline_length_statistics(df: pd.DataFrame) -> pd.Series:
    """Compute statistics for headline lengths.

    Returns a ``pd.Series`` produced by ``describe()`` on lengths.
    """
    lengths = df.get("headline", pd.Series(dtype="object")).astype(str).apply(len)
    return lengths.describe()


def plot_headline_length(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Save distribution plot of headline lengths to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = df.get("headline", pd.Series(dtype="object")).astype(str).apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=50, kde=True, color="blue")
    plt.title("Distribution of Headline Lengths")
    plt.xlabel("Headline Length (Characters)")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "headline_length_distribution.png")
    plt.close()


def publisher_counts(df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """Return the top ``top_n`` publisher counts."""
    return df.get("publisher", pd.Series(dtype="object")).value_counts().head(top_n)


# Additional plotting helpers can be implemented similarly
