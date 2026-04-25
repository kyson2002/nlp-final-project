# print_themes.py
# ThemeDrift
#
# Prints theme buckets with member companies and top words.
# Pass constrained=True/False to match whichever clustering run you want to inspect.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from boilerplate import get_boilerplate

import re

FILINGS_DIR = Path("data/edgar/filings")
METHODS     = ["tfidf", "sbert", "e5"]
YEARS       = [2019, 2021, 2023]
TOP_WORDS   = 10


BOILERPLATE = get_boilerplate("combined")  # use both manual and LLM lists


def load_texts():
    records = {}
    for path in sorted(FILINGS_DIR.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        year   = int(parts[1])
        text   = path.read_text(encoding="utf-8", errors="ignore")
        text   = text.lower()
        text   = re.sub(r"\d+", " ", text)
        text   = re.sub(r"[^a-z\s]", " ", text)
        text   = re.sub(r"\s+", " ", text).strip()
        records[(ticker, year)] = text
    return records


def top_words_for_cluster(members, year, all_texts, constrained, n=TOP_WORDS):
    year_items   = [(k, v) for k, v in all_texts.items() if k[1] == year]
    year_tickers = [k[0] for k, v in year_items]
    year_texts   = [v for k, v in year_items]

    if len(year_texts) < 2:
        return []

    stop = list(TfidfVectorizer(stop_words="english").get_stop_words())
    if constrained:
        stop = stop + BOILERPLATE

    vec = TfidfVectorizer(max_features=5000, stop_words=stop, min_df=2, ngram_range=(1, 2))

    try:
        matrix = vec.fit_transform(year_texts)
    except Exception:
        return []

    vocab      = vec.get_feature_names_out()
    member_set = set(m.upper() for m in members)
    member_idx = [i for i, t in enumerate(year_tickers) if t in member_set]

    if not member_idx:
        return []

    avg_scores = np.asarray(matrix[member_idx].mean(axis=0)).flatten()
    top_idx    = avg_scores.argsort()[::-1][:n]
    return [vocab[i] for i in top_idx]


def print_themes(constrained=True):
    cluster_dir = Path("data/clusters") / ("constrained" if constrained else "free")
    label       = "constrained" if constrained else "free"
    all_texts   = load_texts()

    print(f"\nTheme buckets  [{label}]")
    print(f"Loaded {len(all_texts)} filings\n")

    for method in METHODS:
        for year in YEARS:
            path = cluster_dir / f"assignments_{method}_{year}.csv"
            if not path.exists():
                continue

            df       = pd.read_csv(path)
            n_themes = df[df["cluster"] != -1]["cluster"].nunique()
            n_noise  = (df["cluster"] == -1).sum()

            print(f"\n{'='*60}")
            print(f"  {method.upper()} {year}  |  {n_themes} themes  |  {n_noise} noise  |  [{label}]")
            print(f"{'='*60}")

            for cluster_id in sorted(df["cluster"].unique()):
                members = sorted(df[df["cluster"] == cluster_id]["ticker"].tolist())
                if cluster_id == -1:
                    print(f"\n  noise : {', '.join(members)}")
                    continue

                words = top_words_for_cluster(members, year, all_texts, constrained)
                print(f"\n  theme {cluster_id} ({len(members)} companies)")
                print(f"  companies : {', '.join(members)}")
                print(f"  top words : {', '.join(words) if words else 'none found'}")


# default to constrained - change to False to see the unconstrained version
print_themes(constrained=True)
