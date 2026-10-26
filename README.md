# E-Commerce Review Mining & Automotive Warranty NLP Analytics

An end-to-end Machine Learning and Natural Language Processing engineering project combining dynamic e-commerce review extraction, aspect-based sentiment analysis, automotive warranty text mining, and unsupervised failure-mode clustering.

Built and verified in Python 3.15.1 with complete offline reproducibility, automated reporting pipelines, and modular architecture.

---

## Project Overview

This repository implements two production-grade data intelligence pipelines:

1. **E-Commerce Web Scraping & Aspect Sentiment Intelligence**:
   - Automated JavaScript-rendered scraping of Best Buy Canada product reviews using Selenium WebDriver paired with BeautifulSoup4.
   - Built-in anti-scraping mitigation (stealth headers, request throttling, exponential backoff retry logic, and resilient endpoint fallbacks).
   - VADER polarity scoring and multi-aspect categorization (e.g., `Good Design & Quality (Pos)`, `Comfort & Fit (Neg)`, `Sound Quality (Pos)`) to uncover core customer satisfaction drivers and product pain points.

2. **Automotive Warranty Text Mining & Root-Cause Clustering**:
   - In-depth Exploratory Data Analysis (EDA), automated data profiling, and timeline feature engineering on 1,000 automotive warranty records.
   - Bilingual (Spanish/English) regex & keyword entity extraction capturing components, failure symptoms, diagnostic trouble codes (DTC/SPS), and redacted numeric codes.
   - Unsupervised TF-IDF vectorization and K-Means clustering ($K=5$) with centroid analysis to map free-text narratives into actionable engineering root causes.

---

## Project Deliverables & Artifacts

| Deliverable | Format | Path |
| --- | --- | --- |
| Task 1 Notebook (Executed & Annotated) | `.ipynb` | `notebooks/Task1_BestBuy_Scraping_and_Sentiment.ipynb` |
| Task 1 Notebook (Interactive HTML) | `.html` | `outputs/html_notebooks/Task1_BestBuy_Scraping_and_Sentiment.html` |
| Task 2 Notebook (Executed & Annotated) | `.ipynb` | `notebooks/Task2_EDA_TextMining_and_RootCause.ipynb` |
| Task 2 Notebook (Interactive HTML) | `.html` | `outputs/html_notebooks/Task2_EDA_TextMining_and_RootCause.html` |
| Task 1 Dataset (Scraped + Sentiment Tags) | `.csv` | `outputs/task1_bestbuy_reviews_sentiment.csv` |
| Task 1 Dataset (Scraped + Sentiment Tags) | `.xlsx` | `outputs/task1_bestbuy_reviews_sentiment.xlsx` |
| Task 2 Dataset (Mined Entities + Clusters) | `.csv` | `outputs/task2_tagged_complaints.csv` |
| Task 2 Dataset (Mined Entities + Clusters) | `.xlsx` | `outputs/task2_tagged_complaints.xlsx` |
| Task 1 Technical & Business Report | `.docx` | `outputs/Task1_BestBuy_Report.docx` |
| Task 1 Technical & Business Report | `.pdf` | `outputs/Task1_BestBuy_Report.pdf` |
| Task 2 Stakeholder Executive Report | `.docx` | `outputs/Task2_Stakeholder_Report.docx` |
| Task 2 Stakeholder Executive Report | `.pdf` | `outputs/Task2_Stakeholder_Report.pdf` |
| Data Profiling & Statistical Summary | `.csv` | `outputs/task2_column_profile.csv` |
| Cluster Characteristics & Top Terms | `.csv` | `outputs/task2_cluster_summary.csv` |
| High-Resolution Visualizations | `.png` | `outputs/figures/` (16 plots) |

---

## Quick Start & Pipeline Execution

The entire pipeline runs out-of-the-box from the project root:

```bash
# Run the full pipeline: datasets, Word/PDF reports, and executed notebooks
.venv/Scripts/python.exe run_pipeline.py

# Re-run the live Selenium scraping demonstration against Best Buy Canada
.venv/Scripts/python.exe run_pipeline.py --live

# Fast iteration (generate datasets, tables, and reports without re-executing notebooks)
.venv/Scripts/python.exe run_pipeline.py --skip-notebooks

# Run the automated test suite (20 unit tests)
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

---

## Project Architecture

```
hanuai/
  scrape.py         Selenium + BeautifulSoup scraper, retries, CAPTCHA handling, API fallback
  graded.py         Deterministic review loader for offline benchmarking
  sentiment.py      VADER polarity scoring + multi-aspect tagging
  task1_viz.py      Task 1 rating and sentiment distribution charts
  task2/
    eda.py          Data assessment, imputation, timeline engineering, frequency plots
    text_mining.py  Entity extraction, Spanish normalisation, redacted-code detection
    clustering.py   TF-IDF + K-Means with empty-cluster reinitialization & NumPy fallback
    analysis.py     Temporal/category trend analysis and root-cause mapping
  reporting.py      python-docx report builder & automated PDF compilation
  notebooks.py      Jupyter notebook builder + headless execution engine
  pipeline.py       End-to-end workflow orchestration
run_pipeline.py     Command-line entry point
export_pdfs.ps1     Automated Word-to-PDF export script
```

---

## Dataset & Technical Notes

- **Task 1 Review Data**: Contains 121 analyzed reviews across two benchmark products: 100 live-scraped reviews for the Sony MDR-ZX110 (SKU 10325154) and 21 reviews for the JBL Tune 670NC (SKU 18470223). A snapshot is preserved in `data/demo_reviews_live.csv` for 100% offline reproducibility.
- **Reproducibility**: All stochastic steps (K-Means clustering, dataset sampling) use a fixed seed (`42`).
- **Resilience**: If scikit-learn is unavailable on newer Python builds, the clustering engine seamlessly falls back to a self-contained NumPy TF-IDF + K-Means implementation with empty-cluster reinitialization.

