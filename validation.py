# validation.py
# ThemeDrift
#
# Validates discovered clusters against thematic ETF baskets.
# Two parts:
#   A - Static overlap: precision and recall of clusters vs ETF holdings
#   B - Temporal lead: did our clusters predict ETF membership early?
#
# ETF note: we use genuinely thematic ETFs (cross-sector, narrative-driven)
# not sector/subsector ETFs like XLF or XLV. The point is to test whether
# NLP recovers investment narratives, not GICS classifications.
# Holdings are approximate historical proxies - ETFs are imperfect validators
# by design (liquidity requirements force inclusion of off-theme names).
#
# Outputs:
#   data/validation/overlap_{method}_{year}.csv   - precision/recall per cluster
#   data/validation/lead_time_{method}.csv        - temporal lead time per company
#   data/validation/summary.csv                   - top-line results

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

CLUSTER_DIR    = Path("data/clusters/constrained")
VALIDATION_DIR = Path("data/validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["tfidf", "sbert", "e5"]
YEARS   = [2019, 2021, 2023]


#~~
# ETF baskets - historical proxies per year
# These are thematic ETFs, not sector ETFs. Holdings are approximate snapshots
# based on publicly available historical composition data.
# We focus on names that overlap with our S&P 500 top-50 universe.
#~~

ETF_BASKETS = {

    # AI / cloud / software
    "IGV_software": {
        # iShares Expanded Tech-Software ETF - pure software, cross-sector
        2019: ["MSFT", "ADBE", "CRM", "ORCL", "INTU", "IBM"],
        2021: ["MSFT", "ADBE", "CRM", "ORCL", "INTU", "IBM"],
        2023: ["MSFT", "ADBE", "CRM", "ORCL", "INTU", "IBM"],
    },

    "BOTZ_ai_robotics": {
        # Global X Robotics & AI ETF - AI, automation, robotics
        2019: ["NVDA", "IBM"],
        2021: ["NVDA", "IBM", "GOOGL"],
        2023: ["NVDA", "GOOGL", "MSFT", "AMD"],
    },

    "WCLD_cloud": {
        # WisdomTree Cloud Computing ETF
        2019: ["MSFT", "AMZN", "GOOGL", "CRM", "ORCL"],
        2021: ["MSFT", "AMZN", "GOOGL", "CRM", "ORCL", "IBM"],
        2023: ["MSFT", "AMZN", "GOOGL", "CRM", "ORCL", "IBM", "ADBE"],
    },

    # Fintech / payments
    "FINX_fintech": {
        # Global X FinTech ETF - digital payments, financial technology
        2019: ["V", "MA", "SPGI"],
        2021: ["V", "MA", "SPGI", "INTU"],
        2023: ["V", "MA", "SPGI", "INTU"],
    },

    # Clean energy / energy transition
    "ICLN_clean_energy": {
        # iShares Global Clean Energy ETF
        2019: ["NEE"],
        2021: ["NEE"],
        2023: ["NEE", "LIN"],
    },

    # Genomics / biotech / healthcare innovation
    "ARKG_genomics": {
        # ARK Genomic Revolution ETF
        2019: ["ABBV", "TMO"],
        2021: ["ABBV", "TMO", "LLY"],
        2023: ["ABBV", "LLY", "TMO", "UNH"],
    },

    # Disruptive / innovation
    "ARKK_innovation": {
        # ARK Innovation ETF - disruptive technology across sectors
        2019: ["TSLA", "NVDA", "AMZN"],
        2021: ["TSLA", "GOOGL", "AMZN", "NVDA"],
        2023: ["TSLA", "META", "GOOGL", "AMZN", "NVDA", "MSFT"],
    },

    # Semiconductors (thematic, not sector - AI infrastructure angle)
    "SOXX_semis": {
        # iShares Semiconductor ETF
        2019: ["NVDA", "AMD", "QCOM", "AVGO", "TXN"],
        2021: ["NVDA", "AMD", "QCOM", "AVGO", "TXN"],
        2023: ["NVDA", "AMD", "QCOM", "AVGO", "TXN"],
    },

    # Cybersecurity
    "HACK_cybersecurity": {
        # ETFMG Prime Cyber Security ETF
        2019: ["CSCO", "IBM"],
        2021: ["CSCO", "IBM", "MSFT"],
        2023: ["CSCO", "IBM", "MSFT", "GOOGL"],
    },

}

# live holdings pulled from yfinance for 2023 anchor
# maps our ETF names to their actual ticker for the API call
ETF_LIVE_TICKERS = {
    "IGV_software":     "IGV",
    "BOTZ_ai_robotics": "BOTZ",
    "WCLD_cloud":       "WCLD",
    "FINX_fintech":     "FINX",
    "ICLN_clean_energy":"ICLN",
    "ARKG_genomics":    "ARKG",
    "ARKK_innovation":  "ARKK",
    "SOXX_semis":       "SOXX",
    "HACK_cybersecurity":"HACK",
}


#~~
# Optionally fetch live 2023 ETF holdings to cross-check hardcoded baskets
#~~

def fetch_live_holdings(our_universe, verbose=True):
    """
    Pull current ETF holdings via yfinance and filter to our universe.
    Used as a 2023 anchor to sanity-check the hardcoded baskets.
    Saves to data/validation/live_holdings.csv.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed - skipping live holdings fetch")
        return {}

    live = {}
    for etf_name, ticker in ETF_LIVE_TICKERS.items():
        try:
            holdings = yf.Ticker(ticker).funds_data.top_holdings
            if holdings is None or holdings.empty:
                continue
            # filter to names in our universe
            matched = [t for t in holdings.index if t in our_universe]
            live[etf_name] = matched
            if verbose:
                log.info(f"  {etf_name}: {len(matched)} holdings in our universe -> {matched}")
        except Exception as e:
            log.debug(f"  Could not fetch {ticker}: {e}")

    if live:
        pd.DataFrame([
            {"etf": k, "holdings": ", ".join(v)} for k, v in live.items()
        ]).to_csv(VALIDATION_DIR / "live_holdings.csv", index=False)

    return live


#~~
# Part A - static overlap: precision and recall per cluster
#~~

def compute_overlap(cluster_members, etf_members):
    """
    Precision = fraction of our cluster that is in the ETF
    Recall    = fraction of the ETF that is in our cluster
    F1        = harmonic mean
    """
    cluster_set = set(cluster_members)
    etf_set     = set(etf_members)
    overlap     = cluster_set & etf_set

    precision = len(overlap) / len(cluster_set) if cluster_set else 0
    recall    = len(overlap) / len(etf_set)     if etf_set    else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return round(precision, 3), round(recall, 3), round(f1, 3), sorted(overlap)


def run_overlap(method, year, force=False):
    out_path = VALIDATION_DIR / f"overlap_{method}_{year}.csv"
    if out_path.exists() and not force:
        log.info(f"  Loading cached overlap: {method} {year}")
        return pd.read_csv(out_path)

    assign_path = CLUSTER_DIR / f"assignments_{method}_{year}.csv"
    if not assign_path.exists():
        log.warning(f"  No assignments found for {method} {year} - run clustering first")
        return pd.DataFrame()

    assignments = pd.read_csv(assign_path)
    rows = []

    for cluster_id in sorted(assignments["cluster"].unique()):
        if cluster_id == -1:
            continue
        members = assignments[assignments["cluster"] == cluster_id]["ticker"].tolist()

        best_f1   = 0
        best_row  = None

        for etf_name, baskets in ETF_BASKETS.items():
            etf_members = baskets.get(year, [])
            if not etf_members:
                continue

            precision, recall, f1, overlap = compute_overlap(members, etf_members)

            row = {
                "method":     method,
                "year":       year,
                "cluster_id": cluster_id,
                "n_members":  len(members),
                "etf":        etf_name,
                "etf_size":   len(etf_members),
                "overlap_n":  len(overlap),
                "overlap":    ", ".join(overlap),
                "precision":  precision,
                "recall":     recall,
                "f1":         f1,
                "members":    ", ".join(sorted(members)),
            }
            rows.append(row)

            if f1 > best_f1:
                best_f1  = f1
                best_row = row

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


#~~
# Part B - temporal lead: did our clusters predict ETF membership early?
#~~

def run_temporal_lead(method, force=False):
    """
    For each company that appears in an ETF basket in 2021 or 2023,
    check whether our model already placed them in a matching cluster
    in an earlier year snapshot.

    Lead time = earliest year we clustered them correctly minus
                the year they entered the ETF basket.
    Positive lead = we detected them before the ETF.
    """
    out_path = VALIDATION_DIR / f"lead_time_{method}.csv"
    if out_path.exists() and not force:
        log.info(f"  Loading cached lead time: {method}")
        return pd.read_csv(out_path)

    # load cluster assignments for all years
    clusters = {}
    for year in YEARS:
        path = CLUSTER_DIR / f"assignments_{method}_{year}.csv"
        if path.exists():
            clusters[year] = pd.read_csv(path)

    if len(clusters) < 2:
        log.warning(f"  Need at least 2 years of clusters for {method}")
        return pd.DataFrame()

    rows = []

    for etf_name, baskets in ETF_BASKETS.items():
        # for each company that enters this ETF basket, find when we first detected them
        all_etf_years = sorted(baskets.keys())

        for entry_year in [2021, 2023]:
            if entry_year not in baskets:
                continue

            etf_members = set(baskets[entry_year])
            prev_years  = [y for y in all_etf_years if y < entry_year]

            for company in etf_members:
                # find what cluster this company was in at entry_year
                if entry_year not in clusters:
                    continue
                entry_assign = clusters[entry_year]
                entry_cluster = entry_assign[entry_assign["ticker"] == company]["cluster"].values
                if len(entry_cluster) == 0 or entry_cluster[0] == -1:
                    continue
                entry_cluster_id = entry_cluster[0]

                # find the other members of that cluster at entry_year
                cluster_peers = set(
                    entry_assign[entry_assign["cluster"] == entry_cluster_id]["ticker"].tolist()
                )

                # now check earlier years - was this company already in a cluster
                # that overlapped with the same ETF?
                earliest_detection = None
                for prev_year in prev_years:
                    if prev_year not in clusters:
                        continue
                    prev_assign = clusters[prev_year]
                    prev_row    = prev_assign[prev_assign["ticker"] == company]
                    if prev_row.empty or prev_row["cluster"].values[0] == -1:
                        continue

                    prev_cluster_id = prev_row["cluster"].values[0]
                    prev_members    = set(
                        prev_assign[prev_assign["cluster"] == prev_cluster_id]["ticker"].tolist()
                    )

                    # check if the prev cluster overlaps meaningfully with the ETF basket
                    prev_etf = set(baskets.get(prev_year, baskets.get(entry_year, [])))
                    overlap  = prev_members & prev_etf
                    if len(overlap) >= 2:  # at least 2 ETF members in the cluster
                        earliest_detection = prev_year
                        break

                lead_years = (entry_year - earliest_detection) if earliest_detection else None

                rows.append({
                    "method":             method,
                    "company":            company,
                    "etf":                etf_name,
                    "etf_entry_year":     entry_year,
                    "earliest_detection": earliest_detection,
                    "lead_years":         lead_years,
                    "detected_early":     lead_years is not None and lead_years > 0,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


#~~
# Summary
#~~

def build_summary(overlap_results, lead_results):
    rows = []

    # Part A summary - best ETF match per cluster per method/year
    for (method, year), df in overlap_results.items():
        if df.empty:
            continue
        best = df.sort_values("f1", ascending=False).groupby("cluster_id").first().reset_index()
        for _, row in best.iterrows():
            if row["f1"] > 0:
                rows.append({
                    "part":       "A_overlap",
                    "method":     method,
                    "year":       year,
                    "cluster_id": row["cluster_id"],
                    "best_etf":   row["etf"],
                    "f1":         row["f1"],
                    "precision":  row["precision"],
                    "recall":     row["recall"],
                    "overlap":    row["overlap"],
                    "members":    row["members"],
                })

    # Part B summary - avg lead time per method
    for method, df in lead_results.items():
        if df.empty:
            continue
        detected = df[df["detected_early"] == True]
        if detected.empty:
            continue
        rows.append({
            "part":             "B_temporal",
            "method":           method,
            "year":             "",
            "n_early":          len(detected),
            "avg_lead_years":   round(detected["lead_years"].mean(), 2),
            "companies":        ", ".join(sorted(detected["company"].unique())),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(VALIDATION_DIR / "summary.csv", index=False)
    return summary


#~~
# Print results
#~~

def print_results(overlap_results, lead_results):
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")

    print("\nPart A - ETF Overlap (best match per cluster, F1 > 0.2 shown)")
    print("-"*60)
    for (method, year), df in overlap_results.items():
        if df.empty:
            continue
        best = (df[df["f1"] > 0.2]
                .sort_values("f1", ascending=False)
                .groupby("cluster_id").first()
                .reset_index())
        if best.empty:
            continue
        print(f"\n  {method.upper()} {year}")
        for _, row in best.iterrows():
            print(f"    cluster {int(row['cluster_id']):<3}  {row['etf']:<25}"
                  f"  F1={row['f1']:.2f}  P={row['precision']:.2f}  R={row['recall']:.2f}"
                  f"  overlap: {row['overlap']}")

    print(f"\n\nPart B - Temporal Lead (companies detected before ETF entry)")
    print("-"*60)
    for method, df in lead_results.items():
        if df.empty:
            continue
        detected = df[df["detected_early"] == True]
        if detected.empty:
            print(f"\n  {method.upper()}: no early detections found")
            continue
        avg_lead = detected["lead_years"].mean()
        print(f"\n  {method.upper()}  |  {len(detected)} early detections  |  avg lead {avg_lead:.1f} years")
        for _, row in detected.sort_values("lead_years", ascending=False).iterrows():
            print(f"    {row['company']:<8}  {row['etf']:<25}"
                  f"  detected {row['earliest_detection']}  ->  ETF entry {row['etf_entry_year']}"
                  f"  ({int(row['lead_years'])}yr lead)")

    print(f"\n{'='*60}\n")


#~~
# Main
#~~

def run(force=False, fetch_live=False):
    our_universe = set()
    for baskets in ETF_BASKETS.values():
        for members in baskets.values():
            our_universe.update(members)

    if fetch_live:
        log.info("Fetching live ETF holdings ...")
        live = fetch_live_holdings(our_universe)

    overlap_results = {}
    lead_results    = {}

    log.info("Running Part A - ETF overlap ...")
    for method in METHODS:
        for year in YEARS:
            df = run_overlap(method, year, force=force)
            overlap_results[(method, year)] = df

    log.info("Running Part B - temporal lead ...")
    for method in METHODS:
        df = run_temporal_lead(method, force=force)
        lead_results[method] = df

    summary = build_summary(overlap_results, lead_results)
    print_results(overlap_results, lead_results)

    return overlap_results, lead_results, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
