"""End-to-end orchestration for the two assessment tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import subprocess

from .config import (
    ROOT,
    TASK1_DATASET,
    TASK1_DATASET_XLSX,
    TASK1_REPORT,
    TASK1_REPORT_PDF,
    TASK2_CLUSTERS,
    TASK2_DATASET,
    TASK2_DATASET_XLSX,
    TASK2_PROFILE,
    TASK2_REPORT,
    TASK2_REPORT_PDF,
    ensure_dirs,
    seed_everything,
)
from .graded import graded_metadata, load_graded_reviews
from .reporting import build_task1_report, build_task2_report
from .scrape import get_demo_reviews
from .sentiment import analyze_reviews, summarize_sentiment
from .task1_viz import plot_rating_distribution, plot_task1_figures
from .task2.analysis import run_analysis
from .task2.clustering import cluster_records
from .task2.eda import run_eda
from .task2.text_mining import mine_dataframe, mining_summary

TASK1_EXPORT_COLUMNS = [
    "PrimaryKey",
    "ReviewID",
    "ProductID",
    "ProductName",
    "ReviewTitle",
    "ReviewText",
    "ReviewDate",
    "Rating",
    "SourcePlatform",
    "ReviewerName",
    "VerifiedPurchaser",
    "Recommended",
    "SortFilter",
    "CollectionMethod",
    "Dataset",
    "Polarity",
    "CompoundScore",
    "Aspect_Tag",
    "AspectCount",
    "PrimaryAspect",
    "CleanText",
]

TASK2_INTERNAL_COLUMNS = (
    "Components",
    "Symptoms",
    "ErrorCodes",
    "RedactedCodeContexts",
    "IssueCategories",
    "NormalisedVerbatim",
)


def task1_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    export = frame.copy()
    for column in ("SortFilter", "CollectionMethod"):
        if column not in export:
            export[column] = ""
        export[column] = export[column].fillna("").astype("string")

    graded = export["Dataset"].astype("string").str.startswith("Graded")
    export.loc[graded & (export["SortFilter"] == ""), "SortFilter"] = "Offline recon (no live sort)"
    export.loc[graded & (export["CollectionMethod"] == ""), "CollectionMethod"] = "offline recon snapshot"
    export["SortFilter"] = export["SortFilter"].replace("", "Most Helpful")
    export["CollectionMethod"] = export["CollectionMethod"].replace("", "public review endpoint")

    export["AspectTags"] = export["AspectTags"].map(lambda values: "|".join(values))
    columns = [column for column in TASK1_EXPORT_COLUMNS if column in export]
    return export[columns].reset_index(drop=True)


def task2_export_frame(mined: pd.DataFrame) -> pd.DataFrame:
    export = mined.drop(columns=[c for c in TASK2_INTERNAL_COLUMNS if c in mined], errors="ignore")
    export["IssueCategoriesText"] = mined["IssueCategories"].map(lambda values: "|".join(values))
    if "Cluster" not in export and "Cluster" in mined:
        export["Cluster"] = mined["Cluster"]
    for column in ("ComponentsText", "SymptomsText", "ErrorCodesText"):
        export[column] = export[column].fillna("").replace("", "Not detected")
    export["RedactedCodeContextsText"] = (
        export["RedactedCodeContextsText"].fillna("").replace("", "Not detected")
    )
    export["EntityTags"] = export["EntityTags"].fillna("").replace("", "None detected")
    return export.reset_index(drop=True)


def run_task1(live: bool = False, save: bool = True) -> dict[str, Any]:
    """Run the Task 1 scraping, graded loading and sentiment workflow."""
    ensure_dirs()
    seed_everything()

    graded = analyze_reviews(load_graded_reviews())
    demo_raw, demo_stats = get_demo_reviews(live=live)
    demo = analyze_reviews(demo_raw)

    combined = pd.concat([graded, demo], ignore_index=True, sort=False)
    export = task1_export_frame(combined)

    if save:
        export.to_csv(TASK1_DATASET, index=False, encoding="utf-8")
        try:
            export.to_excel(TASK1_DATASET_XLSX, index=False, engine="openpyxl")
        except Exception:
            pass

    graded_summary = summarize_sentiment(graded)
    demo_summary = summarize_sentiment(demo)
    figures = plot_task1_figures(graded_summary, demo_summary)
    figures["ratings"] = plot_rating_distribution(graded, demo)

    return {
        "graded": graded,
        "demo": demo,
        "combined": combined,
        "export": export,
        "graded_summary": graded_summary,
        "demo_summary": demo_summary,
        "demo_stats": demo_stats,
        "graded_meta": graded_metadata(),
        "figures": figures,
        "dataset_path": TASK1_DATASET,
        "dataset_path_xlsx": TASK1_DATASET_XLSX,
        "rows": int(len(export)),
    }


def run_task2(
    n_clusters: int = 5,
    make_figures: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the Task 2 EDA, mining, clustering and root-cause workflow."""
    ensure_dirs()
    seed_everything()

    eda = run_eda(make_figures=make_figures)
    mined = mine_dataframe(eda["frame"])
    clustering = cluster_records(mined, n_clusters=n_clusters)
    if clustering.frame is not None:
        mined = clustering.frame
    stats = mining_summary(mined)
    analysis = run_analysis(mined, clustering)
    export = task2_export_frame(mined)

    if save:
        export.to_csv(TASK2_DATASET, index=False, encoding="utf-8")
        try:
            export.to_excel(TASK2_DATASET_XLSX, index=False, engine="openpyxl")
        except Exception:
            pass
        eda["profile"].to_csv(TASK2_PROFILE, index=False, encoding="utf-8")
        clustering.summary.to_csv(TASK2_CLUSTERS, index=False, encoding="utf-8")

    return {
        "eda": eda,
        "mined": mined,
        "export": export,
        "clustering": clustering,
        "mining_stats": stats,
        "analysis": analysis,
        "dataset_path": TASK2_DATASET,
        "dataset_path_xlsx": TASK2_DATASET_XLSX,
        "rows": int(len(export)),
    }


def run_reports(task1: dict[str, Any], task2: dict[str, Any]) -> dict[str, Path]:
    """Build both Word and PDF reports from the pipeline results."""
    task1_path = build_task1_report(
        graded=task1["graded"],
        demo=task1["demo"],
        graded_summary=task1["graded_summary"],
        demo_summary=task1["demo_summary"],
        demo_stats=task1["demo_stats"],
        graded_meta=task1["graded_meta"],
        figures=task1.get("figures"),
        output_path=TASK1_REPORT,
    )
    task2_path = build_task2_report(
        eda=task2["eda"],
        mined=task2["mined"],
        mining_stats=task2["mining_stats"],
        clustering=task2["clustering"],
        analysis=task2["analysis"],
        output_path=TASK2_REPORT,
    )

    reports: dict[str, Path] = {
        "task1_report_docx": task1_path,
        "task2_report_docx": task2_path,
    }

    # Generate PDF versions using Word COM if available
    export_pdfs_script = ROOT / "export_pdfs.ps1"
    if export_pdfs_script.exists():
        try:
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(export_pdfs_script)],
                capture_output=True,
                check=False,
                timeout=60,
            )
            if TASK1_REPORT_PDF.exists():
                reports["task1_report_pdf"] = TASK1_REPORT_PDF
            if TASK2_REPORT_PDF.exists():
                reports["task2_report_pdf"] = TASK2_REPORT_PDF
        except Exception:
            pass

    return reports


def run_all(
    live: bool = False,
    n_clusters: int = 5,
    build_notebooks: bool = True,
    make_reports: bool = True,
) -> dict[str, Any]:
    """Run both tasks, optionally generating reports and notebooks."""
    ensure_dirs()
    seed_everything()

    task1 = run_task1(live=live)
    task2 = run_task2(n_clusters=n_clusters)
    reports = run_reports(task1, task2) if make_reports else {}

    notebooks = {}
    if build_notebooks:
        from .notebooks import build_and_execute_notebooks

        notebooks = build_and_execute_notebooks()

    return {
        "task1": task1,
        "task2": task2,
        "reports": reports,
        "notebooks": notebooks,
    }
