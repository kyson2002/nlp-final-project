# print_themes.py
# ThemeDrift
#
# Prints theme buckets with member companies and top words.
# Pass constrained=True/False to match whichever clustering run you want to inspect.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re

FILINGS_DIR = Path("data/edgar/filings")
METHODS     = ["tfidf", "sbert", "e5"]
YEARS       = [2019, 2021, 2023]
TOP_WORDS   = 10

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
