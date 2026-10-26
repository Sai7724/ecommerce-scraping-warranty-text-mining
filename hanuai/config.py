"""Shared configuration for the assessment solution."""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECON_DIR = ROOT / "_recon"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
NOTEBOOK_DIR = ROOT / "notebooks"
TASK2_CSV = ROOT / "Hanuai_task2_assignment.csv"

# A single seed drives every stochastic step (K-Means, any sampling).
RANDOM_SEED = 42

# VADER compound scores inside +/- this band are treated as neutral.
NEUTRAL_BAND = 0.05

# Structured columns whose non-null fraction is below this threshold are
# reported but excluded from frequency-distribution plots.
SPARSE_NON_NULL_THRESHOLD = 0.50

# The assignment brief asks for at least 50 reviews.
MIN_REVIEWS_BRIEF = 50

# Live demonstration product (Sony MDR-ZX110, 11,656 reviews on the site).
DEMO_PRODUCT = {
    "sku": "10325154",
    "name": "Sony MDR-ZX110 Over-Ear Headphones - Black",
    "url": (
        "https://www.bestbuy.ca/en-ca/product/"
        "sony-mdr-zx110-on-ear-headphones-black/10325154"
    ),
}

# Offline graded product (JBL Tune 670NC); data captured in ``_recon/``.
GRADED_PRODUCT = {
    "sku": "18470223",
    "name": "JBL Tune 670NC On-Ear Noise Cancelling Bluetooth Headphones - Blue",
    "url": (
        "https://www.bestbuy.ca/en-ca/product/"
        "jbl-tune-670nc-on-ear-noise-cancelling-bluetooth-headphones-blue/18470223"
    ),
}

REVIEW_SORTS = ("Most Helpful", "Newest", "Highest Rating", "Lowest Rating")

ISSUE_CATEGORIES = (
    "Component Failure",
    "Electrical Issue",
    "Software Glitch",
    "Assembly Quality",
)

VERBATIM_COLUMNS = ("CAUSAL_VERBATIM", "CORRECTION_VERBATIM", "CUSTOMER_VERBATIM")

SENTIMENT_POLARITIES = ("Positive", "Neutral", "Negative")

TASK1_DATASET = OUTPUT_DIR / "task1_bestbuy_reviews_sentiment.csv"
TASK1_DATASET_XLSX = OUTPUT_DIR / "task1_bestbuy_reviews_sentiment.xlsx"
TASK1_DEMO_SNAPSHOT = DATA_DIR / "demo_reviews_live.csv"
TASK1_DEMO_STATS = DATA_DIR / "demo_scrape_stats.json"
TASK2_DATASET = OUTPUT_DIR / "task2_tagged_complaints.csv"
TASK2_DATASET_XLSX = OUTPUT_DIR / "task2_tagged_complaints.xlsx"
TASK2_PROFILE = OUTPUT_DIR / "task2_column_profile.csv"
TASK2_CLUSTERS = OUTPUT_DIR / "task2_cluster_summary.csv"
TASK1_REPORT = OUTPUT_DIR / "Task1_BestBuy_Report.docx"
TASK1_REPORT_PDF = OUTPUT_DIR / "Task1_BestBuy_Report.pdf"
TASK2_REPORT = OUTPUT_DIR / "Task2_Stakeholder_Report.docx"
TASK2_REPORT_PDF = OUTPUT_DIR / "Task2_Stakeholder_Report.pdf"
TASK1_NOTEBOOK = NOTEBOOK_DIR / "Task1_BestBuy_Scraping_and_Sentiment.ipynb"
TASK2_NOTEBOOK = NOTEBOOK_DIR / "Task2_EDA_TextMining_and_RootCause.ipynb"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, NOTEBOOK_DIR):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed the standard-library and NumPy RNGs for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
