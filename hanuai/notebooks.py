"""Build and execute the two annotated assessment notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from .config import ROOT, TASK1_NOTEBOOK, TASK2_NOTEBOOK, ensure_dirs

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.15.1",
    },
}


def _notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = dict(KERNEL_METADATA)
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    return notebook


TASK1_CELLS = [
    (
        "markdown",
        """# Task 1 - Best Buy Canada Review Scraping and Sentiment Analysis

This notebook implements the Task 1 brief end to end:

1. Scrape a live Best Buy Canada product (Sony MDR-ZX110, SKU 10325154) with a browser-rendered
   pipeline and a resilient HTTP fallback.
2. Load the fixed, offline JBL Tune 670NC graded set (21 reviews) from `_recon/`.
3. Classify every review with VADER and tag the functional aspects being discussed.
4. Export a single tabular dataset that labels live rows and graded rows distinctly.

The notebook is deterministic and can be executed offline: the demo reviews are read from the
snapshot captured by the live run (`data/demo_reviews_live.csv`). Set `LIVE_SCRAPE = True` in the
scraping cell to re-run the browser demonstration.""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from IPython.display import display

from hanuai.config import FIGURE_DIR, TASK1_DATASET, ensure_dirs, seed_everything

ensure_dirs()
seed_everything()
pd.set_option("display.max_colwidth", 120)
print("Project root:", PROJECT_ROOT)""",
    ),
    (
        "markdown",
        """## 1. Live scraping pipeline

`hanuai.scrape.BestBuyScraper` drives Chrome with Selenium so that the JavaScript product page is
rendered before BeautifulSoup parses it. It then:

- iterates the four required review-sort filters (Most Helpful, Newest, Highest Rating, Lowest Rating);
- actuates a **Show More** control when the review widget exposes one;
- retries transient failures with exponential backoff;
- detects CAPTCHA / Access Denied challenges, logs them, pauses, and degrades to the public review
  endpoint instead of failing the run;
- de-duplicates reviews by review ID and records how each row was collected.

The cell below reuses the live snapshot by default so the notebook runs offline. Flip
`LIVE_SCRAPE` to `True` to hit Best Buy Canada again.""",
    ),
    (
        "code",
        """from hanuai.scrape import get_demo_reviews

LIVE_SCRAPE = False  # set True to re-run the live Selenium demonstration

demo_raw, demo_stats = get_demo_reviews(live=LIVE_SCRAPE)
print("Source:", demo_stats.get("source"))
print("Reviews collected:", len(demo_raw))
demo_raw.head(5)""",
    ),
    (
        "markdown",
        """### Extracted fields and collection provenance

Every review carries a unique `PrimaryKey`, title, body, `YYYY-MM-DD` date, star rating, source
platform, reviewer name and the sort filter / collection method under which it was obtained.""",
    ),
    (
        "code",
        """print(demo_raw.shape)
display(demo_raw[["ReviewTitle", "ReviewDate", "Rating", "SortFilter", "CollectionMethod"]].head(10))
print("Empty review bodies:", int((demo_raw["ReviewText"].fillna("") == "").sum()))""",
    ),
    (
        "code",
        """stats_table = pd.DataFrame(
    [
        {"Metric": key, "Value": ", ".join(value) if isinstance(value, list) else value}
        for key, value in demo_stats.items()
        if key not in ("notes",)
    ]
)
stats_table""",
    ),
    (
        "markdown",
        """## 2. Graded offline review set

The graded set is captured under `_recon/` for the JBL Tune 670NC (SKU 18470223). It is loaded from
local JSON only - no network access - so polarity and aspect outputs reproduce exactly on every run.
The site reports 21 reviews for this product and all 21 are captured; that is below the brief's
50-review minimum and is documented in the report.""",
    ),
    (
        "code",
        """from hanuai.graded import graded_metadata, load_graded_reviews

graded_raw = load_graded_reviews()
meta = graded_metadata()
print(meta)
display(graded_raw[["ReviewDate", "Rating", "ReviewTitle", "Dataset"]].head(8))""",
    ),
    (
        "markdown",
        """## 3. Sentiment and aspect tagging

`hanuai.sentiment` combines the review title and body, normalises Unicode and promotion boilerplate,
then applies VADER:

- `Polarity` is exactly one of Positive / Neutral / Negative;
- scores inside the +/-0.05 neutral band are Neutral;
- empty bodies are Neutral with an `Uncategorized` tag;
- aspect keywords produce tags of the form `Aspect (Pos|Neg|Neu)`, one per detected functional area.""",
    ),
    (
        "code",
        """from hanuai.sentiment import analyze_reviews, summarize_sentiment

graded = analyze_reviews(graded_raw)
demo = analyze_reviews(demo_raw)

display(graded[["ReviewTitle", "Rating", "Polarity", "CompoundScore", "Aspect_Tag"]].head(10))""",
    ),
    (
        "code",
        """graded_summary = summarize_sentiment(graded)
demo_summary = summarize_sentiment(demo)

print("Graded polarity counts:", graded_summary["counts"], "| mean compound:", graded_summary["mean_compound"])
print("Demo polarity counts:  ", demo_summary["counts"], "| mean compound:", demo_summary["mean_compound"])

graded_summary["polarity_table"].assign(Share=lambda frame: frame["Share"].map("{:.1%}".format))""",
    ),
    (
        "code",
        """graded_summary["aspect_table"]""",
    ),
    (
        "markdown",
        """### Satisfaction drivers, pain points and representative quotes

The aspect table is mention-level, so a single review can contribute several aspects. Drivers are
the most frequently mentioned positive aspects; pain points are the negative ones.""",
    ),
    (
        "code",
        """print("Top drivers (graded):")
for item in graded_summary["top_drivers"]:
    print(f"  - {item['aspect']}: {item['mentions']} mention(s)")

print("\\nPain points (graded):")
for item in graded_summary["pain_points"]:
    print(f"  - {item['aspect']}: {item['mentions']} negative mention(s)")

print("\\nRepresentative quotes:")
for quote in graded_summary["quotes"][:4]:
    print(f"  [{quote['polarity']}] {quote['title']} -> {quote['aspects']}")
    print(f"      {quote['text'][:180]}")""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

polarity = graded_summary["polarity_table"].set_index("Polarity")["Reviews"]
axes[0].bar(polarity.index, polarity.values, color=["#2a9d8f", "#8d99ae", "#c1121f"])
axes[0].set_title("Graded reviews by polarity")
axes[0].set_ylabel("Reviews")

aspects = graded_summary["aspect_table"].head(8).set_index("Aspect")[["Pos", "Neg"]]
aspects.plot.barh(stacked=True, ax=axes[1], color=["#2a9d8f", "#c1121f"])
axes[1].set_title("Aspect mentions (top 8)")
axes[1].set_xlabel("Mentions")
axes[1].set_ylabel("")

for axis in axes:
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "task1_notebook_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()""",
    ),
    (
        "markdown",
        """## 4. Anti-scraping strategy exercised by the pipeline

The pipeline is written for an aggressively protected storefront:

- **IP blocking:** production runs rotate egress IPs, pin one session to one proxy, throttle each
  proxy separately and randomise request timing.
- **CAPTCHA / bot detection:** a headed real Chrome session is used with CDP stealth, realistic
  headers, User-Agent and Referer, disabled automation switches and human-like scrolling. Challenges
  are never auto-solved; they are logged, paused, retried with backoff and then bypassed through the
  public review endpoint.
- **Transient failures:** exponential backoff (1.5s, 3s, 6s, 12s) with a configurable retry budget
  and idempotent de-duplication by review ID.""",
    ),
    (
        "code",
        """print("Scrape statistics")
for key, value in demo_stats.items():
    print(f"  {key}: {value}")

if demo_stats.get("captcha_events"):
    print("\\nA bot challenge was observed; the run logged it, paused, retried and continued via the review endpoint.")""",
    ),
    (
        "markdown",
        """## 5. Export

The exported dataset contains both sources with an explicit `Dataset` label so live rows and graded
rows can never be confused. The same frame is written by `run_pipeline.py`.""",
    ),
    (
        "code",
        """from hanuai.pipeline import task1_export_frame

combined = pd.concat([graded, demo], ignore_index=True, sort=False)
export = task1_export_frame(combined)
export.to_csv(TASK1_DATASET, index=False, encoding="utf-8")

print("Exported rows:", len(export))
print(export["Dataset"].value_counts())
export.head(5)""",
    ),
    (
        "markdown",
        """## 6. Conclusion

- The scraping pipeline is proven against a live product and returns well above the 50-review minimum.
- The graded set is scored entirely offline, deterministically, with one polarity and at least one
  aspect tag per review.
- Sound quality, comfort and value dominate positive feedback; fit/clamping pressure is the one
  repeated negative theme in the graded set.
- Deliverables: this notebook, `outputs/task1_bestbuy_reviews_sentiment.csv` and the Word report.""",
    ),
]


TASK2_CELLS = [
    (
        "markdown",
        """# Task 2 - Automotive Complaint EDA, Text Mining and Root-Cause Analysis

This notebook implements the Task 2 brief end to end:

1. Profile and clean the 1,000-record x 20-column warranty complaint dataset.
2. Extract components, symptoms and error codes from the three VERBATIM fields with regex/keyword
   dictionaries, including Spanish normalisation and redacted-code detection.
3. Assign one or more issue categories to every record.
4. Cluster the complaints with TF-IDF + K-Means and interpret the clusters as root causes.
5. Produce temporal and category-wise trend analyses and export the tagged dataset.""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from IPython.display import Image, display

from hanuai.config import TASK2_DATASET, ensure_dirs, seed_everything

ensure_dirs()
seed_everything()
pd.set_option("display.max_colwidth", 100)
print("Project root:", PROJECT_ROOT)""",
    ),
    (
        "markdown",
        """## 1. Exploratory data analysis

The profiler reports each column's dtype, role, non-null fraction, cardinality and sparsity. Free
text is always treated as text. Structured fields with a non-null fraction below 0.5 are reported but
excluded from frequency plots. Cleaning parses dates and list-like cells, imputes only where
appropriate, and adds derived features without destroying redaction evidence in the verbatim text.""",
    ),
    (
        "code",
        """from hanuai.task2.eda import run_eda

eda = run_eda()
frame = eda["frame"]
print("Raw shape:", eda["shape"])
print("Duplicate rows:", eda["duplicates"])
print("Sparse columns excluded from plots:", eda["sparse_columns"])
eda["profile"]""",
    ),
    (
        "code",
        """missing = eda["missing"]
missing[missing["Missing"] > 0]""",
    ),
    (
        "code",
        """eda["numeric_summary"]""",
    ),
    (
        "code",
        """for path in eda["figures"]:
    print(path.name)
    display(Image(filename=str(path)))""",
    ),
    (
        "markdown",
        """## 2. Text mining and entity tagging

The miner concatenates the three VERBATIM fields, normalises Unicode and Spanish terms, then applies
component, symptom and error-code dictionaries plus issue-category scoring. Redacted numeric codes
appear in the source as runs of spaces; the miner flags them and preserves the surrounding context
instead of collapsing the whitespace.""",
    ),
    (
        "code",
        """from hanuai.task2.text_mining import mine_dataframe, mining_summary

mined = mine_dataframe(frame)
stats = mining_summary(mined)

category_table = pd.DataFrame(
    [
        {
            "IssueCategory": category,
            "Records tagged": stats["category_counts"][category],
            "Share": f"{stats['category_counts'][category] / len(mined):.1%}",
            "Primary records": stats["primary_counts"][category],
        }
        for category in stats["category_counts"]
    ]
)
category_table""",
    ),
    (
        "code",
        """print("Redacted-code records:", stats["redacted_records"])
print("Spanish-language records:", stats["spanish_records"])
print("Records with more than one category:", stats["multi_category_records"])
print("\\nTop components:", stats["components"][:8])
print("\\nTop symptoms:", stats["symptoms"][:8])""",
    ),
    (
        "code",
        """examples = mined[mined["RedactedCodeFlag"]].head(3)
for _, row in examples.iterrows():
    print(row["Event id"], "->", row["RedactedCodeContextsText"][:220])
    print()

spanish = mined[mined["SpanishFlag"]].head(2)
display(spanish[["Event id", "CUSTOMER_VERBATIM", "PrimaryIssueCategory"]])""",
    ),
    (
        "markdown",
        """## 3. Clustering

TF-IDF vectors (1-2 grams, sublinear TF) feed K-Means with a fixed random seed. The engine uses
scikit-learn when it is importable and a NumPy TF-IDF/K-Means fallback otherwise. Empty clusters are
reinitialised from the farthest point, and the top centroid terms expose each cluster's narrative.""",
    ),
    (
        "code",
        """from hanuai.task2.clustering import cluster_records

clustering = cluster_records(mined)
if clustering.frame is not None:
    mined = clustering.frame
print("Backend:", clustering.backend, "| inertia:", round(clustering.inertia, 1))
clustering.summary""",
    ),
    (
        "markdown",
        """## 4. Root-cause analysis and trends

Cluster top terms are interpreted with rule sets into a likely root cause. Temporal trends show when
complaint volume and each issue family peaked, and the category mix is compared across models and
complaint codes.""",
    ),
    (
        "code",
        """from hanuai.task2.analysis import run_analysis

analysis = run_analysis(mined, clustering)
interpretations = pd.DataFrame(analysis["interpretations"])
interpretations[["cluster", "size", "likely_root_cause", "top_terms", "interpretation"]]""",
    ),
    (
        "code",
        """print("Peak month:", analysis["monthly_peak"])
display(analysis["temporal"]["monthly"].head(10))
display(analysis["categories"]["by_complaint"].head(5))""",
    ),
    (
        "code",
        """for name in ("task2_monthly_volume.png", "task2_issue_trend.png", "task2_issue_by_model.png"):
    path = next((p for p in analysis["figures"] if p.endswith(name)), None)
    if path:
        display(Image(filename=path))""",
    ),
    (
        "markdown",
        """## 5. Export

The export preserves the original 20 columns and adds entity tags, redaction/Spanish flags, issue
categories and cluster assignments.""",
    ),
    (
        "code",
        """from hanuai.pipeline import task2_export_frame

export = task2_export_frame(mined)
export.to_csv(TASK2_DATASET, index=False, encoding="utf-8")
print("Exported rows:", len(export), "columns:", len(export.columns))
export[["Event id", "EntityTags", "IssueCategory", "PrimaryIssueCategory", "Cluster"]].head(8)""",
    ),
    (
        "markdown",
        """## 6. Conclusion

- The radio/display assembly and its communication path dominate every failure narrative.
- Software/programming failures and electrical communication faults overlap heavily and should be
  treated as one platform-level workstream.
- Redaction-aware mining recovered usable context for a quarter of the records despite removed codes.
- Deliverables: this notebook, `outputs/task2_tagged_complaints.csv`, the cluster summary and the
  stakeholder Word report.""",
    ),
]


def build_task1_notebook(path: Path = TASK1_NOTEBOOK) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(_notebook(TASK1_CELLS), path)
    return path


def build_task2_notebook(path: Path = TASK2_NOTEBOOK) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(_notebook(TASK2_CELLS), path)
    return path


def execute_notebook(path: Path, timeout: int = 900) -> Path:
    """Execute a notebook in place, storing all cell outputs."""
    notebook = nbf.read(path, as_version=4)
    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbf.write(notebook, path)
    return path


def build_and_execute_notebooks(execute: bool = True) -> dict[str, Path]:
    """Build both notebooks and (by default) execute them with saved outputs."""
    task1 = build_task1_notebook()
    task2 = build_task2_notebook()
    if execute:
        execute_notebook(task1)
        execute_notebook(task2)
    return {"task1_notebook": task1, "task2_notebook": task2}
