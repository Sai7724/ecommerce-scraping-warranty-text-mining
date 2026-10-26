"""Offline graded review set for Task 1.

The graded set is the fixed collection of reviews captured under ``_recon/``
for the JBL Tune 670NC (product id 18470223).  Nothing here touches the
network: sentiment scoring is reproducible from these JSON snapshots alone.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pandas as pd

from .config import GRADED_PRODUCT, RECON_DIR

PROMO_MARKER = "[This review was collected as part of a promotion.]"
GRADED_DATASET_LABEL = "Graded (offline _recon)"


def _review_files() -> list:
    """Return the recon review snapshots in deterministic page order."""
    files = sorted(RECON_DIR.glob("reviews*.json"))
    if not files:
        raise FileNotFoundError(
            f"No graded review snapshots found in {RECON_DIR}. "
            "Expected reviews*.json captured from BestBuy Canada."
        )
    return files


def _normalise_date(value: str | None) -> str:
    """Format a submission timestamp as ``YYYY-MM-DD`` (dates are kept as-is)."""
    if not value:
        return ""
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return text[:10]


def _split_promotion(comment: str) -> tuple[str, bool]:
    """Separate the promotion boilerplate from the review body."""
    if comment.startswith(PROMO_MARKER):
        return comment[len(PROMO_MARKER) :].strip(), True
    return comment, False


def site_review_total() -> int:
    """Total review count reported by the site for the graded product."""
    for path in _review_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        total = (data.get("RatingSummary") or {}).get("OverallReviewCount")
        if total:
            return int(total)
        if data.get("total"):
            return int(data["total"])
    return 0


def load_graded_reviews() -> pd.DataFrame:
    """Load every recon review, de-duplicated by review id.

    Returns one row per review with the fields required for Task 1.  The
    ``Dataset`` column labels these rows distinctly from live demo rows.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _review_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        for review in data.get("reviews", []):
            review_id = str(review.get("id", "")).strip()
            if not review_id or review_id in seen:
                continue
            seen.add(review_id)
            body, promoted = _split_promotion(str(review.get("comment", "")).strip())
            title = re.sub(r"\s+", " ", str(review.get("title", "")).strip())
            records.append(
                {
                    "ReviewID": review_id,
                    "ProductID": str(
                        review.get("ProductID") or data.get("productId") or GRADED_PRODUCT["sku"]
                    ),
                    "ProductName": GRADED_PRODUCT["name"],
                    "ReviewTitle": title,
                    "ReviewText": body,
                    "ReviewDate": _normalise_date(review.get("submissionTime")),
                    "Rating": int(review.get("rating") or 0),
                    "SourcePlatform": "Best Buy Canada",
                    "ReviewerName": str(review.get("reviewerName") or "Anonymous").strip(),
                    "VerifiedPurchaser": bool(review.get("isVerifiedPurchaser")),
                    "Recommended": review.get("isRecommended"),
                    "PromotionFlag": promoted,
                    "Dataset": GRADED_DATASET_LABEL,
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("The recon review snapshots did not contain any reviews.")

    # Deterministic ordering: newest first, then review id as a tie-breaker.
    frame = frame.sort_values(["ReviewDate", "ReviewID"], ascending=[False, False])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "PrimaryKey", [f"{GRADED_PRODUCT['sku']}-{rid}" for rid in frame["ReviewID"]])
    return frame


def graded_metadata() -> dict[str, Any]:
    """Summary used by the report to document the graded-set shortfall."""
    frame = load_graded_reviews()
    return {
        "product_id": GRADED_PRODUCT["sku"],
        "product_name": GRADED_PRODUCT["name"],
        "site_total": site_review_total(),
        "captured": int(len(frame)),
        "date_range": (frame["ReviewDate"].min(), frame["ReviewDate"].max()),
        "mean_rating": round(float(frame["Rating"].mean()), 2),
        "sources": [path.name for path in _review_files()],
    }
