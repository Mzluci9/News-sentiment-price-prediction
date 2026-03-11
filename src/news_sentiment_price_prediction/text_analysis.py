from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


# ensure required NLTK resources are available when the module is imported
for resource in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def clean_headlines(series: pd.Series, extra_stopwords: Iterable[str] | None = None) -> pd.Series:
    """Return a cleaned version of a series of headlines.

    This removes punctuation, lowercases, and filters stopwords.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    if extra_stopwords:
        stop_words |= set(extra_stopwords)

    def _clean(text: str) -> str:
        tokens = nltk.word_tokenize(text or "")
        words = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]
        return " ".join(words)

    return series.fillna("").astype(str).apply(_clean)


def create_wordcloud(text: str, output_path: Union[str, Path]) -> None:
    """Generate a word cloud from ``text`` and save to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not text.strip():
        raise ValueError("No text provided for word cloud generation.")

    wc = WordCloud(width=800, height=400, background_color="white", max_words=100)
    wc.generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()


def lda_topics(corpus: Iterable[str], n_topics: int = 5, **lda_kwargs) -> list[list[str]]:
    """Run LDA topic modeling on a corpus of documents.

    Returns list of topics (each topic is list of top words).
    """
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    topics: list[list[str]] = []
    if X.shape[0] and X.shape[1]:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, **lda_kwargs)
        lda.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topics.append([feature_names[i] for i in top_indices])
    return topics
