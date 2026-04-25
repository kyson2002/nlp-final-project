
# ThemeDrift

### COS 484 Final Project — Princeton University

ThemeDrift clusters S&P 500 companies by the language in their 10-K filings
(Item 1 — Business Description) and tracks how those clusters shift across
three fiscal year snapshots:  **2019 → 2021 → 2023** . The goal is to test
whether NLP-derived "themes" can anticipate thematic ETF compositions before
the market formally labels them.

---

## Setup

### Option 1 — conda (recommended)

```bash
conda create -n themedrift python=3.11
conda activate themedrift
pip install -r requirements.txt
```

### Option 2 — venv

```bash
python -m venv themedrift
source themedrift/bin/activate      # Mac / Linux
themedrift\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Optional: LLM-generated boilerplate

If you want to use the GPT-powered stopword generator (see `boilerplate.py`),
export your OpenAI API key before running:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Quickstart

Run the full pipeline end to end:

```python
# 1. Pull 10-K filings from SEC EDGAR
from data_pull import run as pull
pull()                          # uses tickers.csv, years 2019/2021/2023

# 2. Compute embeddings (TF-IDF, SBERT, E5)
from embeddings import run as embed
embed()

# 3. Cluster into themes
from clustering import run as cluster
cluster(constrained=True)       # with boilerplate filter (recommended)
cluster(constrained=False)      # without filter (for comparison)

# 4. Validate against thematic ETFs
from validation import run as validate
validate()

# 5. Generate figures
from visualize import run as viz
viz()
```

Or run each step as a standalone script:

```bash
python data_pull.py
python embeddings.py
python clustering.py
python validation.py
python visualize.py
python print_themes.py          # human-readable theme printout
```

---

## File Overview

| File                 | Role                                                                           |
| -------------------- | ------------------------------------------------------------------------------ |
| `tickers.csv`      | Input universe — one ticker per row, drives the entire pipeline               |
| `data_pull.py`     | Step 1 — pulls Item 1 text from SEC EDGAR for each company/year               |
| `embeddings.py`    | Step 2 — converts filing text into vectors (TF-IDF, SBERT, E5)                |
| `clustering.py`    | Step 3 — UMAP + HDBSCAN clustering; produces theme assignments                |
| `validation.py`    | Step 4 — measures cluster quality against thematic ETF baskets                |
| `visualize.py`     | Step 5 — generates all report figures (UMAP grid, drift tracks, ETF heatmaps) |
| `boilerplate.py`   | Shared stopword list; supports manual list, GPT-generated, or combined         |
| `print_themes.py`  | Debug utility — prints each theme's member companies and top keywords         |
| `main.ipynb`       | End-to-end notebook that runs and summarizes the full pipeline                 |
| `data_pull.ipynb`  | Interactive version of the data pull step                                      |
| `report.ipynb`     | Notebook for the written report with figures and analysis                      |
| `requirements.txt` | Python dependencies                                                            |

---

## File Details

### `tickers.csv`

CSV with at minimum a `ticker` column. The default file contains the top 50
S&P 500 companies by market cap. Swap in any list — full S&P 500, Nasdaq 100,
or a custom basket — without changing any code. Helper functions in
`data_pull.py` (`build_sp500_universe`, `build_custom_universe`) can generate
this file automatically.

### `data_pull.py`

Pulls 10-K filings from SEC EDGAR in five steps:

1. Load tickers from CSV
2. Map tickers to EDGAR CIK numbers
3. Build a filings index (handles paginated overflow for large filers like JPM)
4. Download filings and extract Item 1 text using regex anchors; save as `TICKER_YEAR.txt`
5. Run a coverage check

Key config constants at the top: `DEFAULT_YEARS`, `MAX_ITEM1_CHARS`,
`YEAR_DATE_WINDOWS`. Filings are cached on disk so re-runs are fast.

### `embeddings.py`

Three embedding methods, all fit on the same corpus for comparability:

* **TF-IDF** — bag-of-words baseline, reduced to 100 dims via SVD (LSA)
* **SBERT** — `all-MiniLM-L6-v2`, 384-dim semantic embeddings
* **E5** — `intfloat/e5-base-v2`, 768-dim; requires `"passage: "` prefix per the model's training convention

All outputs saved to `data/embeddings/` as CSV. Results are cached; pass
`force=True` to recompute.

### `clustering.py`

Takes embeddings and produces theme assignments:

1. UMAP dimensionality reduction (10D for clustering, 2D for visualization)
2. HDBSCAN with `cluster_selection_method='leaf'` to avoid catch-all clusters
3. Soft clustering via `all_points_membership_vectors` — every company gets a
   0–1 affinity score per theme, not just a hard label

The `constrained` flag controls whether the `BOILERPLATE` stopword list is
applied to TF-IDF before clustering. Run both modes to compare.

Outputs per method per year, saved to `data/clusters/{constrained|free}/`:

* `assignments_{method}_{year}.csv` — hard cluster labels
* `affinity_{method}_{year}.csv` — soft affinity scores
* `umap_{method}_{year}.csv` — 2D UMAP coordinates for plotting

### `validation.py`

Two-part validation against nine thematic ETFs (IGV, BOTZ, WCLD, FINX, ICLN,
ARKG, ARKK, SOXX, HACK):

 **Part A — Static overlap** : precision, recall, and F1 score for each cluster
against each ETF basket at each year snapshot.

 **Part B — Temporal lead** : checks whether a cluster in an earlier year already
grouped companies that only entered an ETF basket in a later year. A positive
lead time means NLP detected the theme before the market labeled it.

Outputs saved to `data/validation/`.

### `visualize.py`

Generates four figures for the report:

* **Fig 1** — 3×3 UMAP grid (method × year)
* **Fig 2** — Drift tracks for selected companies across 2019→2021→2023
* **Fig 3** — ETF overlap F1 heatmaps per method per year
* **Fig 4** — Temporal lead bar chart and bubble scatter

All figures saved to `figures/`. Pass `force=True` to regenerate cached figures.

### `boilerplate.py`

Centralizes the stopword list used in both `clustering.py` and `print_themes.py`.
Three modes selectable via `get_boilerplate(source=...)`:

| Source         | Description                                  |
| -------------- | -------------------------------------------- |
| `"manual"`   | Original hardcoded list (no API key needed)  |
| `"llm"`      | GPT-generated list via OpenAI API            |
| `"combined"` | Union of both — best coverage (recommended) |

LLM results are cached to `data/boilerplate_llm_cache.json`.

### `print_themes.py`

Reads cluster assignments and prints a human-readable summary of each theme:
member companies plus the top 10 TF-IDF keywords that characterize them.
Useful for quickly auditing cluster quality. Change `constrained=True/False`
at the bottom to match whichever clustering run you want to inspect.

---

## Data Directory Layout

```
data/
├── universe/
│   └── universe.csv              # cleaned ticker list
├── edgar/
│   ├── cik_map.json              # ticker → EDGAR CIK mapping
│   ├── filings_index.csv         # one row per company/year with filing URL
│   ├── download_status.csv       # extraction quality log
│   └── filings/
│       └── AAPL_2023.txt         # one file per company per year
├── embeddings/
│   ├── tfidf_embeddings.csv
│   ├── sbert_embeddings.csv
│   ├── e5_embeddings.csv
│   └── metadata.csv
├── clusters/
│   ├── constrained/              # boilerplate-filtered clustering results
│   └── free/                     # unfiltered clustering results
├── validation/
│   ├── overlap_{method}_{year}.csv
│   ├── lead_time_{method}.csv
│   └── summary.csv
└── boilerplate_llm_cache.json    # cached GPT stopword list

figures/
├── umap_grid_constrained.png
├── umap_grid_free.png
├── drift_tracks.png
├── etf_overlap_{method}_{year}.png
└── temporal_lead.png
```

---

## Dependencies

See `requirements.txt`. Key packages:

| Package                   | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `sentence-transformers` | SBERT and E5 embeddings                                      |
| `hdbscan`               | Density-based clustering                                     |
| `umap-learn`            | Dimensionality reduction                                     |
| `scikit-learn`          | TF-IDF, SVD, utilities                                       |
| `yfinance`              | Optional: fetch live ETF holdings for 2023 validation anchor |
| `openai`                | Optional: LLM-generated boilerplate (`boilerplate.py`)     |
