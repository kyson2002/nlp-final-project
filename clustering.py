# clustering.py
# ThemeDrift
#
# Clusters company embeddings into themes using HDBSCAN.
# Runs on all three embedding methods and all three years independently.
#
# Key design choices:
#   - HDBSCAN with cluster_selection_method='leaf' to avoid catch-all clusters.
#   - Soft clustering: every company gets a 0-1 affinity score per theme
#   - UMAP for dimensionality reduction before clustering
#   - constrained flag: when True, strips 10-K boilerplate from TF-IDF before
#     clustering so generic legal language doesnt dominate. Toggle for comparison.
#
# Install before running:
#   pip install hdbscan umap-learn
#
# Outputs (per method/year):
#   data/clusters/{constrained|free}/assignments_{method}_{year}.csv
#   data/clusters/{constrained|free}/affinity_{method}_{year}.csv
#   data/clusters/{constrained|free}/umap_{method}_{year}.csv
#   data/clusters/{constrained|free}/theme_summary.csv

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

log = logging.getLogger(__name__)

EMBED_DIR   = Path("data/embeddings")
FILINGS_DIR = Path("data/edgar/filings")

METHODS = ["tfidf", "sbert", "e5"]
YEARS   = [2019, 2021, 2023]

MIN_CLUSTER_SIZE = 2
MIN_SAMPLES      = 1

UMAP_COMPONENTS = 10
UMAP_2D         = 2

# boilerplate words that appear in almost every 10-K and carry no thematic signal
# FIX: extended with personal names (catherine, jamie, miller, etc.) that caused
# GE+MCD phantom cluster — proxy statement name boilerplate must be excluded
BOILERPLATE = [

    # === 1. legal / filing boilerplate ===
    "item", "items", "statements", "form", "sec",
    "financial", "company", "business", "including",
    "related", "information", "management",
    "annual", "report", "fiscal", "year", "quarter", "period",
    "thereof", "herein", "hereof", "hereby",
    "whereas", "notwithstanding", "accordance",

    # === 2. proxy / governance ===
    "board", "committee",
    "shareholder", "shareholders",
    "stockholder", "stockholders",
    "meeting", "vote", "voting",
    "amendment",

    # === 3. executive / titles ===
    "officer", "director", "president", "vice",
    "chief", "executive", "senior",

    # === 4. proxy / filing keywords ===
    "proxy", "pursuant", "duly", "signed",
    "incorporated", "reference",

    # === 5. signature / registrant block ===
    "registrant", "behalf", "principal", "accounting",
    "executed", "undersigned",
    "signature", "signatures",
    "corporate", "persons", "caused",
    "chairman", "authorized", "indicated",
    "capacity", "capacities",

    # === 6. forward-looking / PSLRA ===
    "forward", "looking",
    "historical",
    "expression", "expressions", "similar",
    "projection", "projections",
    "act", "securities", "reform", "private",
    "meaning", "constitute",

    # === 7. forward-looking verbs ===
    "believe", "believes",
    "expect", "expects",
    "anticipate", "anticipates",
    "plan", "plans",
    "intend", "intends",
    "seek", "seeks",
    "estimate", "estimates",

    # === 8. disclosure / narrative ===
    "relating", "related",
    "involve", "involves",
    "outlook",
    "assumption", "assumptions",
    "disclosure", "disclosures",
    "matter", "matters",

    # === 9. risk / litigation ===
    "litigation", "factor", "factors",
    "development", "developments",
    "expectation", "expectations", "regarding",
    "uncertainty", "uncertainties",
    "risk", "risks",

    # === 10. trademark / naming ===
    "trademark", "trademarks", "name", "names",

    # === 11. months / time ===
    "january", "february", "march", "april",

    # === 12. misc formatting ===
    "page", "pages", "applicable", "xa", "jr", "mr", "ms", "mrs",

    # === 13. personal names (sample, not exhaustive) ===
    "catherine", "jamie", "miller", "james", "john", "robert", "michael",
    "william", "david", "thomas", "richard", "charles", "joseph", "christopher",
    "daniel", "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew",
    "kenneth", "george", "brian", "edward", "kevin", "ronald", "timothy",
    "mary", "patricia", "linda", "barbara", "elizabeth", "jennifer", "maria",
    "susan", "margaret", "dorothy", "lisa", "nancy", "karen", "betty",
    "lloyd", "lawrence", "dean",

    # === filing / signature ===
    "exchange", "corporation",
    "director", "directors",
    "requirement", "requirements",
    "statement",

    # === forward-looking ===
    "future", "futures",
    "result", "results",
    "fact", "facts",
    "caution",
    "subject",
]



#~~
# Load embeddings
#~~

def load_embeddings(method, year):
    path = EMBED_DIR / f"{method}_embeddings.csv"
    if not path.exists():
        raise FileNotFoundError(f"No embeddings found at {path} - run embeddings.py first")

    df      = pd.read_csv(path)
    year_df = df[df["year"] == year].reset_index(drop=True)

    if year_df.empty:
        raise ValueError(f"No data for year {year} in {method} embeddings")

    cols   = [c for c in year_df.columns if c.startswith(method)]
    matrix = year_df[cols].values
    meta   = year_df[["ticker", "year"]].copy()
    return matrix, meta


def load_texts_for_year(year):
    records = {}
    for path in sorted(FILINGS_DIR.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        ticker   = parts[0].upper()
        txt_year = int(parts[1])
        if txt_year != year:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.lower()
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        records[ticker] = text
    return records


# Rebuild TF-IDF with constrained stopwords (only used when constrained=True)
def rebuild_tfidf(year, meta, constrained):
    texts_map = load_texts_for_year(year)
    tickers   = meta["ticker"].tolist()
    texts     = [texts_map.get(t, "") for t in tickers]

    stop = list(TfidfVectorizer(stop_words="english").get_stop_words())
    if constrained:
        stop = stop + BOILERPLATE

    if constrained:
        # physically remove stopwords from text before vectorizing
        # this handles cases where min_df would otherwise keep them in vocabulary
        stop_set = set(stop)
        cleaned  = []
        for text in texts:
            tokens  = text.split()
            filtered = [t for t in tokens if t not in stop_set]
            cleaned.append(" ".join(filtered))
        texts = cleaned

    vec = TfidfVectorizer(
        max_features=10_000,
        stop_words=stop,
        min_df=2,
        max_df=0.80,
        ngram_range=(1, 2)
    )
    matrix = vec.fit_transform(texts).toarray()
    svd     = TruncatedSVD(n_components=min(100, matrix.shape[1] - 1), random_state=42)
    reduced = svd.fit_transform(matrix)
    return reduced


# UMAP reduction
def reduce_umap(matrix, n_components, random_state=42):
    try:
        import umap
    except ImportError:
        raise ImportError("Run: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, len(matrix) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


# HDBSCAN clustering
def run_hdbscan(matrix):
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Run: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    clusterer.fit(matrix)
    return clusterer


def get_soft_scores(clusterer, matrix):
    import hdbscan
    soft = hdbscan.all_points_membership_vectors(clusterer)
    return soft


def label_themes(assignments, meta):
    labels = {}
    for cluster_id in sorted(set(assignments)):
        if cluster_id == -1:
            labels[-1] = "noise"
            continue
        members = meta["ticker"][assignments == cluster_id].tolist()
        labels[cluster_id] = ", ".join(members[:5])
    return labels


# Run clustering for one method and one year
def cluster_one(method, year, constrained, out_dir, force=False):
    assign_path   = out_dir / f"assignments_{method}_{year}.csv"
    affinity_path = out_dir / f"affinity_{method}_{year}.csv"
    umap_path     = out_dir / f"umap_{method}_{year}.csv"

    if assign_path.exists() and not force:
        log.info(f"  Loading cached: {method} {year} ({'constrained' if constrained else 'free'})")
        return pd.read_csv(assign_path), pd.read_csv(affinity_path), pd.read_csv(umap_path)

    log.info(f"  Clustering: {method} {year} ({'constrained' if constrained else 'free'}) ...")

    matrix, meta = load_embeddings(method, year)

    if method == "tfidf" and constrained:
        matrix = rebuild_tfidf(year, meta, constrained=True)
    elif method == "tfidf" and not constrained:
        # free mode: rebuild TF-IDF from raw text WITHOUT boilerplate filter
        # (rather than using the pre-saved SVD embeddings, which were fit on all
        # years together — rebuilding per-year keeps constrained/free comparable)
        matrix = rebuild_tfidf(year, meta, constrained=False)

    reduced  = reduce_umap(matrix, n_components=min(UMAP_COMPONENTS, len(matrix) - 2))
    coords2d = reduce_umap(matrix, n_components=UMAP_2D)

    clusterer   = run_hdbscan(reduced)
    assignments = clusterer.labels_
    soft_scores = get_soft_scores(clusterer, reduced)

    n_themes = len(set(assignments)) - (1 if -1 in assignments else 0)
    n_noise  = (assignments == -1).sum()
    log.info(f"    {n_themes} themes, {n_noise} noise")

    assign_df = meta.copy()
    assign_df["cluster"]     = assignments
    assign_df["is_noise"]    = assignments == -1
    assign_df["theme_label"] = assign_df["cluster"].map(label_themes(assignments, meta))

    if soft_scores.shape[1] > 0:
        affinity_cols = {f"theme_{i}": soft_scores[:, i] for i in range(soft_scores.shape[1])}
        affinity_df   = pd.DataFrame(affinity_cols)
        affinity_df.insert(0, "ticker", meta["ticker"].values)
        affinity_df.insert(1, "year",   meta["year"].values)
        score_cols = [c for c in affinity_df.columns if c.startswith("theme_")]
        affinity_df["best_theme"] = affinity_df[score_cols].idxmax(axis=1)
        affinity_df["best_score"] = affinity_df[score_cols].max(axis=1)
    else:
        affinity_df = meta.copy()
        affinity_df["best_theme"] = assignments
        affinity_df["best_score"] = clusterer.probabilities_

    umap_df = meta.copy()
    umap_df["x"]       = coords2d[:, 0]
    umap_df["y"]       = coords2d[:, 1]
    umap_df["cluster"] = assignments

    assign_df.to_csv(assign_path,     index=False)
    affinity_df.to_csv(affinity_path, index=False)
    umap_df.to_csv(umap_path,         index=False)

    return assign_df, affinity_df, umap_df


# Run across all methods and years
def run_all(methods=METHODS, years=YEARS, constrained=True, force=False):
    out_dir = Path("data/clusters") / ("constrained" if constrained else "free")
    out_dir.mkdir(parents=True, exist_ok=True)

    label   = "constrained" if constrained else "free"
    results = {}
    log.info(f"Running clustering [{label}] for {len(methods)} methods x {len(years)} years ...")

    for method in methods:
        for year in years:
            try:
                assign_df, affinity_df, umap_df = cluster_one(
                    method, year, constrained=constrained, out_dir=out_dir, force=force
                )
                results[(method, year)] = {
                    "assignments": assign_df,
                    "affinity":    affinity_df,
                    "umap":        umap_df,
                }
            except Exception as e:
                log.warning(f"  Failed {method} {year}: {e}")

    return results


# Theme summary
def build_theme_summary(results, constrained):
    out_dir = Path("data/clusters") / ("constrained" if constrained else "free")
    rows = []
    for (method, year), res in results.items():
        assign_df   = res["assignments"]
        affinity_df = res["affinity"]

        for cluster_id in sorted(assign_df["cluster"].unique()):
            if cluster_id == -1:
                continue
            members = assign_df[assign_df["cluster"] == cluster_id]["ticker"].tolist()
            member_scores = affinity_df[affinity_df["ticker"].isin(members)]["best_score"]
            avg_score = member_scores.mean() if not member_scores.empty else 0

            rows.append({
                "method":       method,
                "year":         year,
                "theme_id":     cluster_id,
                "n_members":    len(members),
                "avg_affinity": round(avg_score, 3),
                "members":      ", ".join(sorted(members)),
            })

    summary = pd.DataFrame(rows).sort_values(
        ["method", "year", "avg_affinity"], ascending=[True, True, False]
    ).reset_index(drop=True)

    summary.to_csv(out_dir / "theme_summary.csv", index=False)
    return summary


#~~
# Sanity check
#~~

def sanity_check(results, summary, constrained):
    label = "constrained" if constrained else "free"
    print(f"\n{'='*55}")
    print(f"Clustering check  [{label}]")
    print(f"{'='*55}")
    for (method, year), res in results.items():
        assign   = res["assignments"]
        n_themes = assign[assign["cluster"] != -1]["cluster"].nunique()
        n_noise  = assign["is_noise"].sum()
        print(f"  {method:<6} {year}  ->  {n_themes} themes,  {n_noise} noise")
    print(f"{'='*55}\n")




def run(constrained=True, force=False):
    """
    constrained=True  : strips 10-K boilerplate + personal names before clustering
    constrained=False : uses raw TF-IDF, useful for showing phantom cluster examples
    Pass force=True to recompute even if cached results exist.
    """
    results = run_all(constrained=constrained, force=force)
    summary = build_theme_summary(results, constrained=constrained)
    sanity_check(results, summary, constrained=constrained)
    return results, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    run(constrained=True,  force=True)
    run(constrained=False, force=True)