"""K-Means clustering over TF-IDF vectors of the Task 2 VERBATIM fields.

scikit-learn is used when available.  If it cannot be imported (for example
on a Python version without a compatible wheel), the module falls back to a
self-contained TF-IDF + K-Means implementation built on NumPy, including
empty-cluster reinitialisation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RANDOM_SEED

LOGGER = logging.getLogger("hanuai.clustering")

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")
STOPWORDS = {
    "the", "and", "for", "was", "with", "that", "this", "have", "has", "had",
    "not", "are", "but", "from", "they", "will", "been", "were", "into", "when",
    "then", "than", "also", "can", "could", "would", "should", "any", "all",
    "our", "their", "them", "his", "her", "its", "you", "your", "about", "after",
    "before", "over", "under", "out", "off", "per", "via", "upon", "while",
    "there", "these", "those", "what", "which", "who", "how", "why", "does",
    "did", "doing", "done", "being", "because", "between", "during", "each",
    "other", "some", "such", "only", "same", "very", "just", "now", "new",
    "found", "check", "checked", "vehicle", "customer", "concern", "time",
}


def tokenize(text: str) -> list[str]:
    """Tokenise normalised complaint text into content words."""
    return [token for token in TOKEN_RE.findall(str(text).lower()) if token not in STOPWORDS]


@dataclass
class ClusteringResult:
    """Clustering output with everything needed for reporting."""

    labels: np.ndarray
    top_terms: dict[int, list[str]]
    n_clusters: int
    backend: str
    inertia: float
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame: pd.DataFrame | None = None


def _manual_tfidf(
    texts: list[str], min_df: int = 2, max_features: int = 800
) -> tuple[np.ndarray, list[str]]:
    """Build L2-normalised TF-IDF vectors without scikit-learn."""
    documents = [Counter(tokenize(text)) for text in texts]
    doc_frequency: Counter = Counter()
    for document in documents:
        doc_frequency.update(document.keys())

    vocabulary = [
        term
        for term, count in doc_frequency.most_common()
        if count >= min_df
    ][:max_features]
    if not vocabulary:
        vocabulary = [term for term, _ in doc_frequency.most_common(max_features)]

    index = {term: position for position, term in enumerate(vocabulary)}
    n_docs = len(documents)
    idf = np.array(
        [np.log((1 + n_docs) / (1 + doc_frequency[term])) + 1 for term in vocabulary],
        dtype=float,
    )
    matrix = np.zeros((n_docs, len(vocabulary)), dtype=float)
    for row, document in enumerate(documents):
        total = sum(document.values()) or 1
        for term, count in document.items():
            column = index.get(term)
            if column is not None:
                matrix[row, column] = (count / total) * idf[column]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocabulary


def _kmeans_plus_plus(
    matrix: np.ndarray, n_clusters: int, rng: np.random.RandomState
) -> np.ndarray:
    n_samples = matrix.shape[0]
    chosen = [int(rng.randint(n_samples))]
    for _ in range(1, n_clusters):
        distances = np.min(
            np.linalg.norm(matrix[:, None, :] - matrix[chosen][None, :, :], axis=2), axis=1
        )
        probabilities = distances**2
        total = probabilities.sum()
        if total <= 0:
            chosen.append(int(rng.randint(n_samples)))
            continue
        chosen.append(int(rng.choice(n_samples, p=probabilities / total)))
    return matrix[chosen].copy()


def _manual_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int = RANDOM_SEED,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's K-Means with empty-cluster reinitialisation."""
    rng = np.random.RandomState(seed)
    n_samples = matrix.shape[0]
    k = min(n_clusters, n_samples)
    centroids = _kmeans_plus_plus(matrix, k, rng)
    labels = np.full(n_samples, -1, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        for cluster in range(k):
            members = matrix[new_labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
            else:
                # Reinitialise an empty centroid to the point that is furthest
                # from its assigned centroid, then continue the iteration.
                assigned = distances[np.arange(n_samples), new_labels]
                candidate = int(assigned.argmax())
                centroids[cluster] = matrix[candidate]
                new_labels[candidate] = cluster

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
    inertia = float((distances[np.arange(n_samples), labels] ** 2).sum())
    return labels, centroids, inertia


def _top_terms(centroids: np.ndarray, vocabulary: list[str], limit: int = 12) -> dict[int, list[str]]:
    result = {}
    for cluster in range(centroids.shape[0]):
        weights = centroids[cluster]
        order = np.argsort(weights)[::-1]
        result[cluster] = [vocabulary[index] for index in order[:limit] if weights[index] > 0]
    return result


def cluster_texts(
    texts: list[str],
    n_clusters: int = 5,
    seed: int = RANDOM_SEED,
    prefer_sklearn: bool = True,
) -> ClusteringResult:
    """Cluster complaint texts with K-Means on TF-IDF vectors."""
    texts = [str(text) for text in texts]
    if len(texts) < n_clusters:
        n_clusters = max(1, len(texts))

    backend = "numpy fallback"
    if prefer_sklearn:
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=1200,
                min_df=2,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(texts)
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            labels = model.fit_predict(matrix)
            vocabulary = vectorizer.get_feature_names_out().tolist()
            top_terms = _top_terms(model.cluster_centers_, vocabulary)
            inertia = float(model.inertia_)
            backend = "scikit-learn"

            # Defensive check: re-run with the manual engine if a cluster is
            # somehow empty (which would make its centroid meaningless).
            if len(set(labels.tolist())) < n_clusters:
                raise RuntimeError("scikit-learn K-Means produced an empty cluster")
        except Exception as exc:  # ImportError or empty-cluster safeguard
            LOGGER.warning("scikit-learn clustering unavailable (%s); using NumPy fallback.", exc)
            prefer_sklearn = False

    if not prefer_sklearn:
        matrix, vocabulary = _manual_tfidf(texts)
        labels, centroids, inertia = _manual_kmeans(matrix, n_clusters, seed=seed)
        top_terms = _top_terms(centroids, vocabulary)

    return ClusteringResult(
        labels=np.asarray(labels),
        top_terms=top_terms,
        n_clusters=n_clusters,
        backend=backend,
        inertia=inertia,
    )


def cluster_records(
    mined: pd.DataFrame,
    text_column: str = "NormalisedVerbatim",
    n_clusters: int = 5,
    seed: int = RANDOM_SEED,
) -> ClusteringResult:
    """Cluster records and build a per-cluster summary table."""
    result = cluster_texts(mined[text_column].fillna("").tolist(), n_clusters=n_clusters, seed=seed)
    frame = mined.copy()
    frame["Cluster"] = result.labels
    frame["ClusterTopTerms"] = frame["Cluster"].map(
        lambda cluster: ", ".join(result.top_terms.get(int(cluster), []))
    )

    rows = []
    for cluster in range(result.n_clusters):
        subset = frame[frame["Cluster"] == cluster]
        if subset.empty:
            rows.append(
                {
                    "Cluster": cluster,
                    "Size": 0,
                    "Share": 0.0,
                    "TopTerms": ", ".join(result.top_terms.get(cluster, [])),
                    "DominantIssueCategory": "",
                    "TopModel": "",
                    "ExampleReviewID": "",
                }
            )
            continue
        category_counts = subset["PrimaryIssueCategory"].value_counts()
        model_counts = subset["MODEL"].value_counts()
        rows.append(
            {
                "Cluster": cluster,
                "Size": int(len(subset)),
                "Share": round(len(subset) / len(frame), 3),
                "TopTerms": ", ".join(result.top_terms.get(cluster, [])),
                "DominantIssueCategory": str(category_counts.index[0]) if len(category_counts) else "",
                "TopModel": str(model_counts.index[0]) if len(model_counts) else "",
                "ExampleReviewID": str(subset["Event id"].iloc[0]),
            }
        )
    result.summary = pd.DataFrame(rows)
    result.frame = frame
    return result
