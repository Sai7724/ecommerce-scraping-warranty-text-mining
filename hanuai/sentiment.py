"""Lexicon-based sentiment classification and aspect tagging (Task 1).

VADER supplies the polarity; keyword/rule dictionaries supply aspect tags.
Every review receives exactly one :class:`Polarity` and one or more
``<Aspect> (<Pos|Neg|Neu>)`` tags.  The module is deterministic and never
touches the network.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import NEUTRAL_BAND, SENTIMENT_POLARITIES

_ANALYZER = SentimentIntensityAnalyzer()

_PROMO_RE = re.compile(r"\[\s*this review was collected as part of a promotion\.?\s*\]", re.I)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
_QUALITY_RE = re.compile(
    r"^(?:[^a-zA-Z0-9]+|[0-9]+(?:[.,][0-9]+)?\s*(?:stars?|/5)?[^a-zA-Z0-9]*)+$", re.I
)

# Aspect -> keyword/phrase dictionary.  Keywords are matched on word
# boundaries and allow a simple plural/gerund suffix.
ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Good Design & Quality": (
        "design",
        "build quality",
        "construction",
        "look",
        "looks",
        "stylish",
        "well built",
        "solid",
        "quality",
        "premium",
        "fold",
        "foldable",
        "material",
        "form factor",
        "sleek",
    ),
    "Sound Quality": (
        "sound",
        "sound quality",
        "audio",
        "bass",
        "treble",
        "clarity",
        "soundstage",
        "instrument separation",
        "volume",
        "music",
        "equalizer",
        "eq",
        "loud",
        "distortion",
        "static",
    ),
    "Comfort & Fit": (
        "comfort",
        "comfortable",
        "tight",
        "fit",
        "clamp",
        "headband",
        "ear cup",
        "ear cups",
        "earcup",
        "earcups",
        "wear",
        "painful",
        "pain",
        "glasses",
        "lightweight",
        "light weight",
        "heavy",
        "pressure",
        "on-ear",
        "over-ear",
    ),
    "Value for Money": (
        "price",
        "value",
        "cheap",
        "affordable",
        "deal",
        "worth",
        "expensive",
        "cost",
        "money",
        "bargain",
    ),
    "Battery Life": (
        "battery",
        "charge",
        "charging",
        "playtime",
        "play time",
        "power",
        "runtime",
        "recharge",
    ),
    "Connectivity & Bluetooth": (
        "bluetooth",
        "connect",
        "connection",
        "connectivity",
        "pair",
        "pairing",
        "wireless",
        "wired",
        "cable",
        "signal",
        "usb",
        "aux",
        "jack",
    ),
    "Noise Cancellation": (
        "noise cancellation",
        "noise cancelling",
        "noise canceling",
        "noise reduction",
        "anc",
        "isolation",
        "block out",
        "blocking",
    ),
    "Ease of Use": (
        "easy",
        "simple",
        "controls",
        "control",
        "button",
        "buttons",
        "setup",
        "set up",
        "intuitive",
        "user friendly",
        "instructions",
        "app",
    ),
    "Durability & Build": (
        "durable",
        "durability",
        "broke",
        "broken",
        "crack",
        "cracked",
        "damage",
        "lasted",
        "sturdy",
        "flimsy",
        "fell apart",
        "stopped working",
    ),
    "Delivery & Packaging": (
        "delivery",
        "delivered",
        "shipping",
        "arrived",
        "packaging",
        "package",
        "box",
        "on time",
        "fast shipping",
        "late",
    ),
    "Customer Service": (
        "customer service",
        "support",
        "warranty",
        "return",
        "returned",
        "refund",
        "service",
        "replacement",
    ),
}

_ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(?:" + "|".join(re.escape(keyword) for keyword in keywords) + r")(?:s|es|ed|ing)?\b",
        re.IGNORECASE,
    )
    for aspect, keywords in ASPECT_KEYWORDS.items()
}

_SHORT_POLARITY = {"Positive": "Pos", "Negative": "Neg", "Neutral": "Neu"}


def normalise_text(value: Any) -> str:
    """Normalise a title/body pair for classification.

    Unicode is NFKC-normalised, curly quotes/dashes are folded, promotion
    boilerplate is removed, and whitespace is collapsed.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    text = _PROMO_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def combine_review(title: Any, body: Any) -> str:
    """Combine Review Title and Review Body Text into one normalised string."""
    return normalise_text(f"{normalise_text(title)} {normalise_text(body)}")


def polarity_from_compound(compound: float, band: float = NEUTRAL_BAND) -> str:
    """Map a VADER compound score to Positive/Neutral/Negative."""
    if compound >= band:
        return "Positive"
    if compound <= -band:
        return "Negative"
    return "Neutral"


def extract_aspect_tags(text: str, overall_compound: float) -> list[str]:
    """Return ``<Aspect> (<Pos|Neg|Neu>)`` tags for a normalised review."""
    matched = [aspect for aspect, pattern in _ASPECT_PATTERNS.items() if pattern.search(text)]
    if not matched:
        return ["Uncategorized"]

    tags: list[str] = []
    for aspect in matched:
        pattern = _ASPECT_PATTERNS[aspect]
        scores = []
        sentences = [part for part in _SENTENCE_RE.split(text) if part.strip()] or [text]
        for sentence in sentences:
            if pattern.search(sentence):
                scores.append(_ANALYZER.polarity_scores(sentence)["compound"])
        local = sum(scores) / len(scores) if scores else overall_compound
        label = polarity_from_compound(local)
        tags.append(f"{aspect} ({_SHORT_POLARITY[label]})")
    return tags


def classify_review(title: Any, body: Any) -> dict[str, Any]:
    """Classify a single review title/body pair.

    Empty bodies receive the Neutral polarity and an ``Uncategorized`` tag.
    """
    text = combine_review(title, body)
    if _QUALITY_RE.match(text):
        text = ""

    if not text:
        return {
            "CleanText": "",
            "Polarity": "Neutral",
            "CompoundScore": 0.0,
            "AspectTags": ["Uncategorized"],
            "AspectTagsText": "Uncategorized",
            "PrimaryAspect": "Uncategorized",
            "AspectCount": 1,
        }

    compound = float(_ANALYZER.polarity_scores(text)["compound"])
    polarity = polarity_from_compound(compound)
    tags = extract_aspect_tags(text, compound)
    primary = tags[0].rsplit(" (", 1)[0] if tags else "Uncategorized"
    return {
        "CleanText": text,
        "Polarity": polarity,
        "CompoundScore": round(compound, 4),
        "AspectTags": tags,
        "AspectTagsText": "; ".join(tags),
        "PrimaryAspect": primary,
        "AspectCount": len(tags),
    }


def analyze_reviews(
    reviews: pd.DataFrame,
    title_column: str = "ReviewTitle",
    body_column: str = "ReviewText",
) -> pd.DataFrame:
    """Add sentiment and aspect columns to a review frame.

    The input is not mutated; the returned frame preserves the original
    columns and row order.
    """
    frame = reviews.copy()
    results = [
        classify_review(row.get(title_column, ""), row.get(body_column, ""))
        for _, row in frame.iterrows()
    ]
    result_frame = pd.DataFrame.from_records(results)
    for column in result_frame.columns:
        frame[column] = result_frame[column].values

    # Normalise the tag column to a string for tabular exports while keeping
    # the list form available in-process.
    frame["Aspect_Tag"] = frame["AspectTagsText"]
    return frame


def _explode_aspects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tags in frame["AspectTags"]:
        for tag in tags:
            if " (" in tag and tag.endswith(")"):
                aspect, sentiment = tag.rsplit(" (", 1)
                rows.append({"Aspect": aspect, "Sentiment": sentiment.strip("()")})
            else:
                rows.append({"Aspect": tag, "Sentiment": "Neu"})
    return pd.DataFrame(rows, columns=["Aspect", "Sentiment"])


def summarize_sentiment(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate polarity, aspect, driver and pain-point statistics."""
    total = int(len(frame))
    counts = frame["Polarity"].value_counts().reindex(SENTIMENT_POLARITIES, fill_value=0)
    polarity_table = pd.DataFrame(
        {
            "Polarity": counts.index,
            "Reviews": counts.values,
            "Share": [round(value / total, 3) if total else 0.0 for value in counts.values],
        }
    )

    aspects = _explode_aspects(frame)
    if aspects.empty:
        aspect_table = pd.DataFrame(columns=["Aspect", "Pos", "Neu", "Neg", "Mentions"])
    else:
        pivot = (
            aspects.assign(Value=1)
            .pivot_table(index="Aspect", columns="Sentiment", values="Value", aggfunc="sum", fill_value=0)
            .reindex(columns=["Pos", "Neu", "Neg"], fill_value=0)
        )
        pivot["Mentions"] = pivot.sum(axis=1)
        aspect_table = pivot.sort_values("Mentions", ascending=False).reset_index()

    def top_aspects(sentiment: str, limit: int = 5) -> list[dict[str, Any]]:
        """Rank aspects by the number of mentions with the given sentiment."""
        if aspect_table.empty or sentiment not in aspect_table:
            return []
        subset = aspect_table[aspect_table[sentiment] > 0]
        if subset.empty:
            return []
        ranked = subset.sort_values([sentiment, "Mentions"], ascending=False).head(limit)
        return [
            {"aspect": row["Aspect"], "mentions": int(row[sentiment])}
            for _, row in ranked.iterrows()
        ]

    quotes = []
    for polarity, ascending in (("Positive", False), ("Negative", True)):
        subset = frame[frame["Polarity"] == polarity].sort_values(
            "CompoundScore", ascending=ascending, kind="mergesort"
        )
        for _, row in subset.head(4).iterrows():
            quotes.append(
                {
                    "polarity": polarity,
                    "title": row.get("ReviewTitle", ""),
                    "text": row.get("ReviewText", ""),
                    "aspects": row.get("AspectTagsText", ""),
                    "compound": row.get("CompoundScore", 0.0),
                }
            )

    return {
        "total": total,
        "counts": {name: int(counts[name]) for name in SENTIMENT_POLARITIES},
        "polarity_table": polarity_table,
        "aspect_table": aspect_table,
        "top_drivers": top_aspects("Pos"),
        "pain_points": top_aspects("Neg"),
        "quotes": quotes,
        "mean_compound": round(float(frame["CompoundScore"].mean()), 4) if total else 0.0,
    }
