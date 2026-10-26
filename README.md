# HanuAi ML Assessment

Two-task machine-learning assessment implemented in Python 3.15.1 and run entirely from the
project's `.venv`.

- **Task 1:** Best Buy Canada review scraping, VADER sentiment analysis and aspect tagging.
- **Task 2:** Automotive warranty complaint EDA, entity tagging, TF-IDF/K-Means clustering and
  root-cause analysis.

## Deliverables

| Deliverable | Format | Path |
| --- | --- | --- |
| Task 1 notebook (executed, annotated) | `.ipynb` | `notebooks/Task1_BestBuy_Scraping_and_Sentiment.ipynb` |
| Task 1 notebook (HTML viewer) | `.html` | `outputs/html_notebooks/Task1_BestBuy_Scraping_and_Sentiment.html` |
| Task 2 notebook (executed, annotated) | `.ipynb` | `notebooks/Task2_EDA_TextMining_and_RootCause.ipynb` |
| Task 2 notebook (HTML viewer) | `.html` | `outputs/html_notebooks/Task2_EDA_TextMining_and_RootCause.html` |
| Task 1 dataset (scraped + sentiment tags) | `.csv` | `outputs/task1_bestbuy_reviews_sentiment.csv` |
| Task 1 dataset (scraped + sentiment tags) | `.xlsx` | `outputs/task1_bestbuy_reviews_sentiment.xlsx` |
| Task 2 dataset (mined entities + clusters) | `.csv` | `outputs/task2_tagged_complaints.csv` |
| Task 2 dataset (mined entities + clusters) | `.xlsx` | `outputs/task2_tagged_complaints.xlsx` |
| Task 1 report (written report) | `.docx` | `outputs/Task1_BestBuy_Report.docx` |
| Task 1 report (written report) | `.pdf` | `outputs/Task1_BestBuy_Report.pdf` |
| Task 2 report (stakeholder report) | `.docx` | `outputs/Task2_Stakeholder_Report.docx` |
| Task 2 report (stakeholder report) | `.pdf` | `outputs/Task2_Stakeholder_Report.pdf` |
| Supporting tables | `.csv` | `outputs/task2_column_profile.csv`, `outputs/task2_cluster_summary.csv` |
| High-resolution figures | `.png` | `outputs/figures/` (16 plots) |

## Run

```bash
# Full pipeline: datasets, reports, notebooks (uses the cached live snapshot)
.venv/Scripts/python.exe run_pipeline.py

# Re-run the live Selenium scraping demonstration first
.venv/Scripts/python.exe run_pipeline.py --live

# Faster iterations
.venv/Scripts/python.exe run_pipeline.py --skip-notebooks
.venv/Scripts/python.exe run_pipeline.py --skip-notebooks --skip-reports

# Tests
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Layout

```
hanuai/
  scrape.py         Selenium + BeautifulSoup scraper, retries, CAPTCHA handling, API fallback
  graded.py         Offline 21-review JBL Tune 670NC graded loader
  sentiment.py      VADER polarity + aspect tagging
  task1_viz.py      Task 1 charts
  task2/
    eda.py          Profiling, cleaning, imputation, frequency plots
    text_mining.py  Entity extraction, Spanish normalisation, redacted-code detection
    clustering.py   TF-IDF + K-Means with a NumPy fallback and empty-cluster reinit
    analysis.py     Temporal/category trends and root-cause interpretation
  reporting.py      python-docx report generation
  notebooks.py      Notebook builders + executor
  pipeline.py       End-to-end orchestration
run_pipeline.py     CLI entry point
```

## Data notes

- The graded set is fixed at **21 reviews** captured under `_recon/` for the JBL Tune 670NC
  (SKU 18470223). The product has 21 reviews in total, so the graded set is below the brief's
  50-review minimum; the shortfall is documented in the Task 1 report.
- The live demonstration target is the Sony MDR-ZX110 (SKU 10325154). A snapshot of the live run is
  stored in `data/demo_reviews_live.csv` so the notebooks run offline and reproduce exactly.
- Task 1 rows carry a `Dataset` label distinguishing `Demo (live scrape)` from
  `Graded (offline _recon)`.
- All stochastic steps (K-Means, any sampling) use a fixed seed (`42`).
- If scikit-learn is unavailable on Python 3.15, the clustering engine falls back to a NumPy
  TF-IDF + K-Means implementation.
