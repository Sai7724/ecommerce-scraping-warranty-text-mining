"""Temporal trends, category comparisons and root-cause interpretation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ..config import FIGURE_DIR, ISSUE_CATEGORIES
from .clustering import ClusteringResult

CATEGORY_COLORS = {
    "Component Failure": "#c1121f",
    "Electrical Issue": "#f4a261",
    "Software Glitch": "#2a9d8f",
    "Assembly Quality": "#6a4c93",
}

ROOT_CAUSE_RULES: dict[str, tuple[str, ...]] = {
    "Software Glitch": (
        "software",
        "update",
        "program",
        "programmed",
        "programming",
        "sps",
        "calibration",
        "configuration",
        "usb",
        "ota",
    ),
    "Electrical Issue": (
        "wiring",
        "short",
        "ground",
        "connector",
        "harness",
        "voltage",
        "coax",
        "circuit",
        "battery",
    ),
    "Assembly Quality": ("loose", "rattle", "trim", "panel", "fit", "clip", "squeak"),
    "Component Failure": (
        "blank",
        "fault",
        "dead",
        "broken",
        "failure",
        "replace",
        "replaced",
        "module",
        "camera",
        "screen",
    ),
}

ROOT_CAUSE_TEXT = {
    "Software Glitch": (
        "Software/programming fault: repeated SPS/USB update attempts around "
        "{terms} point to corrupted or unavailable calibration files."
    ),
    "Electrical Issue": (
        "Electrical integrity fault: wiring, connector or ground issues around "
        "{terms} interrupt communication between modules."
    ),
    "Assembly Quality": (
        "Assembly/fitment defect: loose or poorly seated hardware around "
        "{terms} causes intermittent operation."
    ),
    "Component Failure": (
        "Hardware failure: the {terms} component fails internally and is "
        "resolved by replacement."
    ),
}


def _save(fig: plt.Figure, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def temporal_trends(mined: pd.DataFrame) -> dict[str, Any]:
    """Monthly and quarterly complaint volume, overall and by category."""
    frame = mined.copy()
    frame["Month"] = frame["OpenedDate"].dt.to_period("M").astype(str)
    frame["Quarter"] = frame["OpenedDate"].dt.to_period("Q").astype(str)

    monthly = (
        frame.groupby("Month", as_index=False)
        .size()
        .rename(columns={"size": "Complaints"})
        .sort_values("Month")
    )
    monthly_by_category = (
        frame.pivot_table(
            index="Month",
            columns="PrimaryIssueCategory",
            values="Event id",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
        .reindex(columns=list(ISSUE_CATEGORIES), fill_value=0)
        .reset_index()
    )
    quarterly = (
        frame.groupby(["Quarter", "PrimaryIssueCategory"], as_index=False)
        .size()
        .rename(columns={"size": "Complaints"})
    )

    # Month-over-month change for the largest categories.
    trend = monthly_by_category.copy()
    for category in ISSUE_CATEGORIES:
        if category in trend:
            trend[f"{category} MoM"] = trend[category].diff().fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(monthly["Month"], monthly["Complaints"], marker="o", color="#1f77b4", linewidth=2)
    ax.set_title("Monthly Complaint Volume")
    ax.set_xlabel("Month opened")
    ax.set_ylabel("Complaints")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    volume_figure = _save(fig, "task2_monthly_volume.png")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bottom = None
    for category in ISSUE_CATEGORIES:
        values = monthly_by_category.get(category)
        if values is None:
            continue
        ax.bar(
            monthly_by_category["Month"],
            values,
            bottom=bottom,
            label=category,
            color=CATEGORY_COLORS[category],
        )
        bottom = values if bottom is None else bottom + values
    ax.set_title("Issue Categories Over Time")
    ax.set_xlabel("Month opened")
    ax.set_ylabel("Complaints")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    category_figure = _save(fig, "task2_issue_trend.png")

    return {
        "monthly": monthly,
        "monthly_by_category": monthly_by_category,
        "quarterly": quarterly,
        "trend": trend,
        "figures": [volume_figure, category_figure],
    }


def category_trends(mined: pd.DataFrame) -> dict[str, Any]:
    """Category-wise comparisons across models and complaint codes."""
    top_models = mined["MODEL"].value_counts().head(10).index.tolist()
    model_subset = mined[mined["MODEL"].isin(top_models)]
    by_model = pd.crosstab(
        model_subset["MODEL"], model_subset["PrimaryIssueCategory"]
    ).reindex(columns=list(ISSUE_CATEGORIES), fill_value=0)
    by_model = by_model.loc[top_models]

    by_complaint = pd.crosstab(
        mined["COMPLAINT_CD_DESC"].fillna("Unknown"), mined["PrimaryIssueCategory"]
    ).reindex(columns=list(ISSUE_CATEGORIES), fill_value=0)
    by_complaint["Total"] = by_complaint.sum(axis=1)
    by_complaint = by_complaint.sort_values("Total", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(by_model.div(by_model.sum(axis=1), axis=0).values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(by_model.columns)), by_model.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(by_model.index)), by_model.index, fontsize=8)
    ax.set_title("Issue Category Mix by Model (row share)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    model_figure = _save(fig, "task2_issue_by_model.png")

    components: Counter = Counter()
    for values in mined["Components"]:
        components.update(values)
    component_table = pd.DataFrame(components.most_common(12), columns=["Component", "Mentions"])

    symptoms: Counter = Counter()
    for values in mined["Symptoms"]:
        symptoms.update(values)
    symptom_table = pd.DataFrame(symptoms.most_common(12), columns=["Symptom", "Mentions"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    component_table.sort_values("Mentions").plot.barh(
        x="Component", y="Mentions", ax=ax, color="#457b9d", legend=False
    )
    ax.set_title("Most Mentioned Components")
    ax.set_xlabel("Records mentioning component")
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    component_figure = _save(fig, "task2_top_components.png")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    symptom_table.sort_values("Mentions").plot.barh(
        x="Symptom", y="Mentions", ax=ax, color="#e76f51", legend=False
    )
    ax.set_title("Most Frequent Symptoms")
    ax.set_xlabel("Records mentioning symptom")
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    symptom_figure = _save(fig, "task2_top_symptoms.png")

    return {
        "by_model": by_model,
        "by_complaint": by_complaint,
        "component_table": component_table,
        "symptom_table": symptom_table,
        "figures": [model_figure, component_figure, symptom_figure],
    }


def interpret_clusters(clustering: ClusteringResult) -> list[dict[str, Any]]:
    """Interpret each cluster's top terms as a likely root cause."""
    interpretations = []
    for row in clustering.summary.itertuples(index=False):
        terms = [term.strip() for term in str(row.TopTerms).split(",") if term.strip()]
        joined = " ".join(terms)
        scores = {
            category: sum(1 for keyword in keywords if keyword in joined)
            for category, keywords in ROOT_CAUSE_RULES.items()
        }
        likely = max(scores, key=lambda category: scores[category])
        if scores[likely] == 0:
            likely = str(row.DominantIssueCategory) or "Component Failure"

        focus = ", ".join(terms[:5])
        interpretation = ROOT_CAUSE_TEXT.get(likely, "").format(terms=focus)
        interpretations.append(
            {
                "cluster": int(row.Cluster),
                "size": int(row.Size),
                "share": float(row.Share),
                "top_terms": terms,
                "likely_root_cause": likely,
                "interpretation": interpretation,
                "dominant_category": str(row.DominantIssueCategory),
                "top_model": str(row.TopModel),
            }
        )
    return interpretations


def cluster_figure(clustering: ClusteringResult) -> str:
    """Bar chart of cluster sizes."""
    summary = clustering.summary
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"C{row.Cluster}" for row in summary.itertuples(index=False)]
    ax.bar(labels, summary["Size"], color="#264653")
    ax.set_title(f"Complaint Clusters (K={clustering.n_clusters}, {clustering.backend})")
    ax.set_ylabel("Records")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, "task2_cluster_sizes.png")


def run_analysis(mined: pd.DataFrame, clustering: ClusteringResult) -> dict[str, Any]:
    """Run all trend analyses and root-cause interpretation."""
    temporal = temporal_trends(mined)
    categories = category_trends(mined)
    interpretations = interpret_clusters(clustering)
    return {
        "temporal": temporal,
        "categories": categories,
        "interpretations": interpretations,
        "figures": temporal["figures"] + categories["figures"] + [cluster_figure(clustering)],
        "monthly_peak": (
            temporal["monthly"].sort_values("Complaints", ascending=False).iloc[0].to_dict()
            if not temporal["monthly"].empty
            else {}
        ),
    }
