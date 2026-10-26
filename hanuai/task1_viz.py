"""Charts for the Task 1 sentiment and aspect results."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR

POLARITY_COLORS = {
    "Positive": "#2a9d8f",
    "Neutral": "#8d99ae",
    "Negative": "#c1121f",
}


def _save(fig: plt.Figure, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_task1_figures(
    graded_summary: dict[str, Any],
    demo_summary: dict[str, Any],
) -> dict[str, str]:
    """Render the polarity and aspect charts used in the Task 1 report."""
    figures: dict[str, str] = {}

    rows = []
    for label, summary in (("Graded (JBL)", graded_summary), ("Demo (Sony)", demo_summary)):
        table = summary["polarity_table"].set_index("Polarity")
        total = max(int(table["Reviews"].sum()), 1)
        for polarity in ("Positive", "Neutral", "Negative"):
            rows.append(
                {
                    "Dataset": label,
                    "Polarity": polarity,
                    "Share": float(table.loc[polarity, "Reviews"]) / total * 100,
                }
            )
    polarity_frame = pd.DataFrame(rows)
    pivot = polarity_frame.pivot(index="Dataset", columns="Polarity", values="Share").reindex(
        columns=["Positive", "Neutral", "Negative"], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = None
    for polarity in ("Positive", "Neutral", "Negative"):
        values = pivot[polarity]
        ax.barh(pivot.index, values, left=bottom, label=polarity, color=POLARITY_COLORS[polarity])
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("Share of reviews (%)")
    ax.set_title("Polarity mix by dataset")
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    figures["polarity"] = _save(fig, "task1_polarity_by_dataset.png")

    aspect_table = graded_summary["aspect_table"]
    if not aspect_table.empty:
        top = aspect_table.head(10).set_index("Aspect")[["Pos", "Neu", "Neg"]]
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        top.plot.barh(stacked=True, ax=ax, color=["#2a9d8f", "#8d99ae", "#c1121f"])
        ax.set_title("Graded set: aspect mentions with sentiment")
        ax.set_xlabel("Mentions")
        ax.set_ylabel("")
        ax.legend(frameon=False, fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        figures["aspects"] = _save(fig, "task1_aspect_performance.png")

    return figures


def plot_rating_distribution(graded: pd.DataFrame, demo: pd.DataFrame) -> str:
    """Grouped bar chart of star ratings for both datasets."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ratings = [1, 2, 3, 4, 5]
    width = 0.38
    positions = list(range(len(ratings)))
    for offset, (label, frame, color) in enumerate(
        (
            ("Graded (JBL)", graded, "#264653"),
            ("Demo (Sony)", demo, "#e9c46a"),
        )
    ):
        counts = frame["Rating"].value_counts().reindex(ratings, fill_value=0)
        ax.bar([p + offset * width for p in positions], counts.values, width=width, label=label, color=color)
    ax.set_xticks([p + width / 2 for p in positions], [f"{star}" for star in ratings])
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Reviews")
    ax.set_title("Star rating distribution")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, "task1_rating_distribution.png")
