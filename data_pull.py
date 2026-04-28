#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
ThemeDrift — data_pull.py

Pulls 10-K Item 1 text from SEC EDGAR for a given universe of companies.
Default targets fiscal years 2019, 2021, 2023 for temporal analysis,
but years and universe are fully configurable.

The universe is driven by a CSV with at minimum a 'ticker' column.
Swap in any list — top 50 S&P, full S&P 500, Nasdaq 100, custom basket.
Use build_universe() helpers below or just drop in your own CSV.

From Jupyter:
    from data_pull import run
    universe, index_df, status = run()
    universe, index_df, status = run("sp500_full.csv")
    universe, index_df, status = run("my_basket.csv", years=[2020, 2022, 2024])
"""

import re
import time
import json
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


#~~
# Config
#~~

TICKERS_CSV  = "tickers.csv"
DATA_DIR     = Path("data")
UNIVERSE_DIR = DATA_DIR / "universe"
EDGAR_DIR    = DATA_DIR / "edgar"
FILINGS_DIR  = EDGAR_DIR / "filings"

# default snapshot years - override by passing years=[] to run()
DEFAULT_YEARS = [2019, 2021, 2023]

# 10-Ks for a fiscal year typically get filed in the first half of the following year
# extend this dict if you add more years
YEAR_DATE_WINDOWS = {
    2018: ("2019-01-01", "2019-12-31"),
    2019: ("2020-01-01", "2020-12-31"),
    2020: ("2021-01-01", "2021-12-31"),
    2021: ("2022-01-01", "2022-12-31"),
    2022: ("2023-01-01", "2023-12-31"),
    2023: ("2024-01-01", "2024-12-31"),
    2024: ("2025-01-01", "2025-12-31"),
}

EDGAR_HEADERS = {
    "User-Agent": "ThemeDrift research@themedrift.io",
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_RATE_LIMIT = 0.6
EDGAR_TIMEOUT    = 30
MAX_ITEM1_CHARS  = 30_000

ITEM1_START = [
    r"item\s+1[\.\s]+business\b",
    r"item\s+1\s*[\.\-\u2013\u2014]\s*business",
]
ITEM1_END = [
    r"item\s+1a[\.\s]+risk\s+factors",
    r"item\s+1a\s*[\.\-\u2013\u2014]\s*risk",
    r"item\s+2[\.\s]+properties",
]

for d in [UNIVERSE_DIR, EDGAR_DIR, FILINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


#~~
# Universe helpers — use these to build a tickers CSV, or bring your own
#~~

def build_sp500_universe(out_path="sp500_full.csv", force=False):
    """
    Pull the full S&P 500 from Wikipedia and save to CSV.
    Use this if you want to run ThemeDrift on the entire index.
    """
    path = Path(out_path)
    if path.exists() and not force:
        log.info(f"S&P 500 list already exists at {path}")
        return pd.read_csv(path)

    log.info("Fetching S&P 500 from Wikipedia ...")
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0].rename(columns={
        "Symbol": "ticker", "Security": "name", "GICS Sector": "gics_sector",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    df = df[["ticker", "name", "gics_sector"]].copy()
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df)} tickers to {path}")
    return df


def build_custom_universe(tickers, names=None, out_path="custom.csv"):
    """
    Build a universe CSV from a plain list of tickers.

    Example:
        build_custom_universe(["NVDA", "AMD", "INTC", "AVGO"], out_path="semis.csv")
    """
    df = pd.DataFrame({"ticker": [t.upper() for t in tickers]})
    if names:
        df["name"] = names
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)}-ticker universe to {out_path}")
    return df


#~~
# Step 1: load tickers from CSV
#~~

def load_tickers(csv_path=TICKERS_CSV):
    # CSV-driven so we can swap universes without touching the code
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ticker file not found: {path}\n"
            f"Tip: use build_sp500_universe() or build_custom_universe() to create one."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("CSV must have a 'ticker' column.")

    #-- clean tickers
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    df = df.drop_duplicates("ticker").reset_index(drop=True)

    log.info(f"Loaded {len(df)} tickers from {path}")
    df.to_csv(UNIVERSE_DIR / "universe.csv", index=False)
    return df


#~~
# Step 2: map tickers to EDGAR CIKs
#~~

def build_cik_map(tickers, force=False):
    # EDGAR uses CIK numbers internally - pull the full mapping once and cache it
    cache = EDGAR_DIR / "cik_map.json"
    if cache.exists() and not force:
        log.info("Loading CIK map from cache")
        return json.loads(cache.read_text())

    log.info("Fetching CIK map from SEC EDGAR ...")
    resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=EDGAR_HEADERS, timeout=EDGAR_TIMEOUT)
    resp.raise_for_status()

    #-- match against our ticker list
    edgar = pd.DataFrame.from_dict(resp.json(), orient="index")
    edgar["ticker_up"]  = edgar["ticker"].str.upper()
    edgar["cik_padded"] = edgar["cik_str"].astype(str).str.zfill(10)

    wanted  = set(t.upper() for t in tickers)
    matched = edgar[edgar["ticker_up"].isin(wanted)]
    cik_map = dict(zip(matched["ticker_up"], matched["cik_padded"]))

    missing = wanted - set(cik_map)
    if missing:
        log.warning(f"No CIK found for: {sorted(missing)}")

    log.info(f"Matched {len(cik_map)}/{len(tickers)} tickers")
    cache.write_text(json.dumps(cik_map, indent=2))
    return cik_map


#~~
# Step 3: find the right 10-K filing for each company/year
#~~

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

def _fetch_all_tenk(cik, years):
    """
    Fetch all 10-K filings for a company across the recent list + any paginated
    older files. Heavy filers (banks etc.) push old 10-Ks off the recent list
    into numbered overflow pages like CIK000xxx-submissions-001.json.
    We stop fetching pages once we have all target years covered.
    """
    _OLDER_URL = "https://data.sec.gov/submissions/{filename}"

    def _parse_page(data):
        return pd.DataFrame({
            "form": data.get("form", []),
            "date": data.get("filingDate", []),
            "acc":  data.get("accessionNumber", []),
            "doc":  data.get("primaryDocument", []),
        })

    resp = requests.get(_SUBMISSIONS_URL.format(cik=cik),
                        headers=EDGAR_HEADERS, timeout=EDGAR_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()

    #-- start with the recent page
    all_tenk = _parse_page(body.get("filings", {}).get("recent", {}))
    all_tenk = all_tenk[all_tenk["form"] == "10-K"]

    #-- check which target years are still missing
    def years_found(df):
        found = set()
        for year in years:
            start, end = YEAR_DATE_WINDOWS[year]
            if not df[(df["date"] >= start) & (df["date"] <= end)].empty:
                found.add(year)
        return found

    still_needed = set(years) - years_found(all_tenk)

    #-- only page through older files if we are still missing years
    older_files = body.get("filings", {}).get("files", [])
    for page in older_files:
        if not still_needed:
            break
        try:
            r = requests.get(_OLDER_URL.format(filename=page["name"]),
                             headers=EDGAR_HEADERS, timeout=EDGAR_TIMEOUT)
            r.raise_for_status()
            page_tenk = _parse_page(r.json())
            page_tenk = page_tenk[page_tenk["form"] == "10-K"]
            all_tenk  = pd.concat([all_tenk, page_tenk], ignore_index=True)
            still_needed = set(years) - years_found(all_tenk)
            time.sleep(EDGAR_RATE_LIMIT)
        except Exception as e:
            log.debug(f"  Error fetching older page {page['name']}: {e}")

    return all_tenk


def build_filings_index(cik_map, years=DEFAULT_YEARS, force=False):
    index_path = EDGAR_DIR / "filings_index.csv"
    if index_path.exists() and not force:
        log.info("Loading filings index from cache")
        return pd.read_csv(index_path)

    missing_years = [y for y in years if y not in YEAR_DATE_WINDOWS]
    if missing_years:
        raise ValueError(f"No date window defined for: {missing_years}. Add to YEAR_DATE_WINDOWS.")

    log.info(f"Indexing filings for {len(cik_map)} companies x {years} ...")
    records = []

    for i, (ticker, cik) in enumerate(cik_map.items()):
        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(cik_map)} done ...")
        try:
            tenk = _fetch_all_tenk(cik, years)

            #-- match each target year to the filing in its expected date window
            for year in years:
                start, end = YEAR_DATE_WINDOWS[year]
                match = tenk[(tenk["date"] >= start) & (tenk["date"] <= end)]
                if match.empty:
                    continue

                row = match.sort_values("date").iloc[0]
                acc_nodash = row["acc"].replace("-", "")
                records.append({
                    "ticker":      ticker,
                    "year":        year,
                    "cik":         cik,
                    "accession":   row["acc"],
                    "filing_date": row["date"],
                    "doc_url": _DOC_URL.format(cik=int(cik), acc_nodash=acc_nodash, doc=row["doc"]),
                })

        except Exception as e:
            log.warning(f"  Error for {ticker}: {e}")

        time.sleep(EDGAR_RATE_LIMIT)

    df = pd.DataFrame(records).sort_values(["ticker", "year"]).reset_index(drop=True)
    df.to_csv(index_path, index=False)

    for year in years:
        log.info(f"  {year}: {(df['year'] == year).sum()} filings")
    log.info(f"Total: {len(df)} filings, {df['ticker'].nunique()} companies")
    return df


#~~
# Step 4: download filings and extract Item 1 text
#~~

def download_all_texts(index_df, force=False):
    results = []
    log.info(f"Downloading {len(index_df)} filings ...")

    for i, row in index_df.iterrows():
        ticker = row["ticker"]
        year   = int(row["year"])
        out    = FILINGS_DIR / f"{ticker}_{year}.txt"

        #-- skip if already on disk
        if out.exists() and not force:
            text = out.read_text(encoding="utf-8", errors="ignore")
            results.append(_make_row(ticker, year, row["filing_date"], out, text, "cached"))
            continue

        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(index_df)}  {ticker} {year}")

        text, method = _fetch_and_extract(row["doc_url"])

        if text:
            out.write_text(text, encoding="utf-8")
            results.append(_make_row(ticker, year, row["filing_date"], out, text, method))
        else:
            log.warning(f"  failed: {ticker} {year}")
            results.append({
                "ticker": ticker, "year": year, "filing_date": row["filing_date"],
                "text_path": None, "char_count": 0,
                "extraction_method": "failed", "status": "failed"
            })

        time.sleep(EDGAR_RATE_LIMIT)

    status_df = pd.DataFrame(results)
    status_df.to_csv(EDGAR_DIR / "download_status.csv", index=False)

    ok     = status_df["status"].isin(["ok", "cached"]).sum()
    failed = (status_df["status"] == "failed").sum()
    log.info(f"Done - {ok} OK, {failed} failed")
    return status_df


def _fetch_and_extract(url):
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.debug(f"HTTP error: {e}")
        return None, "http_error"

    #-- strip HTML and clean whitespace
    text = re.sub(r"<[^>]+>",     " ", resp.text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;",      " ", text)
    text = re.sub(r"\s+",         " ", text).strip()

    return _extract_item1(text)


def _extract_item1(text):
    # try to isolate just Item 1 using section headers as anchors
    lower = text.lower()

    start = None
    for pat in ITEM1_START:
        m = re.search(pat, lower)
        if m:
            start = m.start()
            break

    end = None
    if start is not None:
        window = lower[start + 50:]
        for pat in ITEM1_END:
            m = re.search(pat, window)
            if m:
                end = start + 50 + m.start()
                break

    #-- label extraction method so we can audit quality later
    if start is not None and end is not None and end > start + 200:
        return text[start:end].strip()[:MAX_ITEM1_CHARS], "item1_extracted"
    elif start is not None:
        return text[start:start + MAX_ITEM1_CHARS].strip(), "item1_start_only"
    else:
        return text[:MAX_ITEM1_CHARS], "full_text_fallback"


def _make_row(ticker, year, date, path, text, method):
    return {
        "ticker": ticker, "year": year, "filing_date": date,
        "text_path": str(path), "char_count": len(text),
        "extraction_method": method,
        "status": "ok" if method != "failed" else "failed",
    }


#~~
# Step 5: coverage check
#~~

def sanity_check(status_df, n_companies, years=DEFAULT_YEARS):
    ok = status_df[status_df["status"].isin(["ok", "cached"])]
    print("\n" + "="*50)
    print("Coverage check")
    print("="*50)
    for year in years:
        n = (ok["year"] == year).sum()
        print(f"  {year}: {n}/{n_companies} companies ({n/n_companies*100:.0f}%)")
    full = (ok.groupby("ticker")["year"].count() == len(years)).sum()
    print(f"\n  All {len(years)} years present: {full}/{n_companies} companies")
    print(f"  Avg Item 1 length:   {ok['char_count'].mean():.0f} chars")
    print("\n  Extraction breakdown:")
    print(status_df["extraction_method"].value_counts().to_string(header=False))
    print("="*50 + "\n")


#~~
# Main
#~~

def run(tickers_csv=TICKERS_CSV, years=DEFAULT_YEARS, force=False):
    """
    Full pipeline. Swap universe or years without touching anything else.

    Examples:
        run()                                        # default top-50, 3 years
        run("sp500_full.csv")                        # full S&P 500
        run("semis.csv", years=[2020, 2021, 2022])   # custom basket, custom years
    """
    universe = load_tickers(tickers_csv)
    tickers  = universe["ticker"].tolist()
    cik_map  = build_cik_map(tickers, force=force)
    index_df = build_filings_index(cik_map, years=years, force=force)
    status   = download_all_texts(index_df, force=force)
    sanity_check(status, len(tickers), years=years)
    return universe, index_df, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=TICKERS_CSV)
    parser.add_argument("--years",   default="2019,2021,2023")
    parser.add_argument("--force",   action="store_true")
    args, _ = parser.parse_known_args()
    years = [int(y) for y in args.years.split(",")]
    run(tickers_csv=args.tickers, years=years, force=args.force)


# **Comments:**
# 
# 
# The data pull is complete. We pulled 10-K filings from SEC EDGAR for 50 of the largest S&P 500 companies, covering three fiscal years: 2019, 2021, and 2023. We have full coverage - 50 companies x 3 years = 150 filings total. The data lives in the data/ folder:
# 
# - data/universe/universe.csv - the list of 50 companies with name and sector
# - data/edgar/filings_index.csv - one row per company/year with the EDGAR filing URL
# - data/edgar/filings/ - 150 text files, one per company per year (e.g. AAPL_2019.txt)
# - data/edgar/cik_map.json - maps tickers to EDGAR internal IDs (CIKs)
# - data/edgar/download_status.csv - extraction quality log for each filing
# 
# 
# What data_pull.py does, step by step:
# 
# Step 1 - Loads the ticker list from tickers.csv. The CSV approach means we can swap in any universe (full S&P 500, Nasdaq 100, custom basket) just by pointing at a different file, without changing any code.
# 
# Step 2 - Maps tickers to CIK numbers. EDGAR does not use tickers internally, it uses CIK identifiers. We fetch the full company-to-CIK mapping from EDGAR once and cache it locally.
# 
# Step 3 - Builds a filings index. For each company and each target year, we find the right 10-K filing on EDGAR. One complication: large financial companies like JPM and BAC file hundreds of forms per year, so their older 10-Ks get pushed off
# the standard API results page into paginated overflow files. The code handles this by fetching additional pages until all target years are found.
# 
# Step 4 - Downloads the filings and extracts Item 1 text. Item 1 (Business Description) is the section of the 10-K where companies describe what they do. We use regex anchors to isolate just that section rather than downloading the entire filing. Each file is saved as TICKER_YEAR.txt and capped at 30,000 characters. The extraction method is logged so we can audit quality.
# 
# Step 5 - Runs a coverage check to confirm how many companies and years were successfully pulled.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




