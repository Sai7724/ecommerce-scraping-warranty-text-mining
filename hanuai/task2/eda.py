"""Exploratory data analysis for the Task 2 complaint dataset."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ..config import FIGURE_DIR, SPARSE_NON_NULL_THRESHOLD, TASK2_CSV

LIST_COLUMNS = (
    "Trigger",
    "Failure Component",
    "Failure Condition",
    "Additional Context",
    "Fix Component",
    "Fix Condition",
)
DATE_COLUMNS = ("Opened date", "BUILD_DATE", "IN_USE_DATE")
VERBATIM_COLUMNS = ("CAUSAL_VERBATIM", "CORRECTION_VERBATIM", "CUSTOMER_VERBATIM")
CATEGORICAL_COLUMNS = (
    "MAKE",
    "MODEL",
    "BUILD_PLANT_DESC",
    "CAUSAL_CD_DESC",
    "COMPLAINT_CD_DESC",
    "PLANT",
)
IDENTIFIER_COLUMNS = ("Event id",)

PLOT_BLUE = "#1f77b4"
PLOT_ORANGE = "#ff7f0e"


def classify_role(column: str) -> str:
    """Infer the analysis role of a column from its name and content."""
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    if column in DATE_COLUMNS:
        return "date"
    if column in VERBATIM_COLUMNS:
        return "text"
    if column in LIST_COLUMNS:
        return "list"
    if column == "MODLYR":
        return "numeric"
    if column in CATEGORICAL_COLUMNS:
        return "categorical"
    return "categorical"


def _parse_list(value: Any) -> list[str]:
    """Parse a Python-literal-looking list cell into a list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed]
    return [str(parsed).strip()]


def load_task2_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Task 2 CSV with all columns as strings where sensible."""
    source = path or TASK2_CSV
    frame = pd.read_csv(source)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string")
    return frame


def build_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile every column: dtype, missingness, cardinality and role."""
    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        fraction = round(non_null / total, 4) if total else 0.0
        role = classify_role(column)
        sample = series.dropna().iloc[0] if non_null else ""
        rows.append(
            {
                "Column": column,
                "Dtype": str(series.dtype),
                "Role": role,
                "NonNull": non_null,
                "Missing": total - non_null,
                "NonNullFraction": fraction,
                "Unique": int(series.nunique(dropna=True)),
                "Sparse": bool(
                    role not in ("identifier", "text")
                    and fraction < SPARSE_NON_NULL_THRESHOLD
                ),
                "Example": str(sample)[:120],
            }
        )
    return pd.DataFrame(rows)


def clean_task2(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean, impute and derive features appropriate to each column type."""
    clean = frame.copy()

    # Trim categorical/identifier values; verbatim text is preserved exactly
    # so redacted numeric codes (runs of spaces) remain detectable.
    for column in list(IDENTIFIER_COLUMNS) + list(CATEGORICAL_COLUMNS):
        if column in clean:
            clean[column] = clean[column].astype("string").str.strip()

    # Parse list-like structured columns.
    for column in LIST_COLUMNS:
        if column in clean:
            clean[column] = clean[column].map(_parse_list)

    # Parse dates into dedicated datetime columns and keep the raw strings.
    for source, target in (
        ("Opened date", "OpenedDate"),
        ("BUILD_DATE", "BuildDate"),
        ("IN_USE_DATE", "InUseDate"),
    ):
        clean[target] = pd.to_datetime(
            clean[source], errors="coerce", format="mixed", utc=True
        ).dt.tz_localize(None)

    # Missingness handling.
    clean["CausalMissingFlag"] = clean["CAUSAL_VERBATIM"].isna()
    clean["CAUSAL_VERBATIM"] = clean["CAUSAL_VERBATIM"].fillna("")
    clean["CAUSAL_CD_DESC"] = clean["CAUSAL_CD_DESC"].fillna("Unknown")
    clean["InUseDateMissingFlag"] = clean["InUseDate"].isna()

    # Derived numeric features.
    clean["DaysBuildToOpen"] = (clean["OpenedDate"] - clean["BuildDate"]).dt.days
    clean["DaysInUseToOpen"] = (clean["OpenedDate"] - clean["InUseDate"]).dt.days
    combined = (
        clean["CAUSAL_VERBATIM"].fillna("")
        + " "
        + clean["CORRECTION_VERBATIM"].fillna("")
        + " "
        + clean["CUSTOMER_VERBATIM"].fillna("")
    )
    clean["VerbatimCharLength"] = combined.str.len()
    clean["VerbatimWordCount"] = combined.str.split().str.len()

    clean["MODLYR"] = pd.to_numeric(clean["MODLYR"], errors="coerce")
    if clean["MODLYR"].isna().any():
        clean["MODLYR"] = clean["MODLYR"].fillna(clean["MODLYR"].median())

    return clean


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/standard deviation for numeric analysis features."""
    columns = [
        "MODLYR",
        "DaysBuildToOpen",
        "DaysInUseToOpen",
        "VerbatimCharLength",
        "VerbatimWordCount",
    ]
    rows = []
    for column in columns:
        if column not in frame:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Feature": column,
                "Count": int(series.count()),
                "Mean": round(float(series.mean()), 2),
                "Median": round(float(series.median()), 2),
                "Std": round(float(series.std()), 2),
                "Min": round(float(series.min()), 2),
                "Max": round(float(series.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Top values for each categorical column."""
    rows = []
    for column in CATEGORICAL_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].dropna().value_counts()
        rows.append(
            {
                "Column": column,
                "Unique": int(values.size),
                "TopValue": str(values.index[0]) if len(values) else "",
                "TopCount": int(values.iloc[0]) if len(values) else 0,
            }
        )
    return pd.DataFrame(rows)


def _save_series_bar(
    series: pd.Series,
    title: str,
    path: Path,
    color: str = PLOT_BLUE,
    top: int = 15,
    xlabel: str = "Records",
) -> Path:
    values = series.dropna().value_counts().head(top)
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.32 * len(values) + 1.2)))
    values.sort_values().plot(kind="barh", color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_frequencies(frame: pd.DataFrame) -> list[Path]:
    """Generate frequency-distribution charts for key categorical features."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    figures.append(
        _save_series_bar(frame["MAKE"], "Complaints by Make", FIGURE_DIR / "task2_make_distribution.png")
    )
    figures.append(
        _save_series_bar(
            frame["MODEL"],
            "Top 15 Models by Complaint Volume",
            FIGURE_DIR / "task2_model_top15.png",
            color=PLOT_ORANGE,
        )
    )
    figures.append(
        _save_series_bar(
            frame["COMPLAINT_CD_DESC"],
            "Complaint Categories",
            FIGURE_DIR / "task2_complaint_categories.png",
            color="#2a9d8f",
        )
    )
    figures.append(
        _save_series_bar(
            frame["CAUSAL_CD_DESC"],
            "Top 15 Causal Codes",
            FIGURE_DIR / "task2_causal_codes.png",
            color="#8d5a97",
        )
    )
    figures.append(
        _save_series_bar(
            frame["BUILD_PLANT_DESC"],
            "Complaints by Build Plant",
            FIGURE_DIR / "task2_plant_distribution.png",
            color="#c45a2c",
        )
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(frame["VerbatimCharLength"].dropna(), bins=40, color=PLOT_BLUE, edgecolor="white")
    ax.set_title("Distribution of Verbatim Text Length")
    ax.set_xlabel("Characters (three verbatim fields combined)")
    ax.set_ylabel("Records")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    length_path = FIGURE_DIR / "task2_verbatim_length.png"
    fig.savefig(length_path, dpi=150)
    plt.close(fig)
    figures.append(length_path)
    return figures


def run_eda(path: Path | None = None, make_figures: bool = True) -> dict[str, Any]:
    """Run the full EDA workflow and return the cleaned frame and summaries."""
    raw = load_task2_csv(path)
    profile = build_profile(raw)
    clean = clean_task2(raw)
    missing = (
        raw.isna()
        .sum()
        .rename("Missing")
        .to_frame()
        .assign(NonNullFraction=lambda table: 1 - table["Missing"] / len(raw))
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    duplicates = int(raw.duplicated().sum())
    sparse_columns = profile.loc[profile["Sparse"], "Column"].tolist()

    result: dict[str, Any] = {
        "raw": raw,
        "frame": clean,
        "profile": profile,
        "missing": missing,
        "numeric_summary": numeric_summary(clean),
        "categorical_summary": categorical_summary(clean),
        "duplicates": duplicates,
        "sparse_columns": sparse_columns,
        "shape": raw.shape,
        "figures": [],
    }
    if make_figures:
        result["figures"] = plot_frequencies(clean)
    return result
