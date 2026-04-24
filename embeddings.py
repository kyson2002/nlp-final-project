# embeddings.py
# ThemeDrift
#
# Converts Item 1 text into numerical vectors for clustering.
# Three methods:
#   - TF-IDF       : bag-of-words baseline, fast and interpretable
#   - Sentence-BERT: general purpose semantic embeddings (all-MiniLM-L6-v2)
#   - E5           : stronger semantic model, better on similarity/clustering tasks
#
# All three use the same sentence-transformers library.
# Install once before running: pip install sentence-transformers
#
# Outputs:
#   data/embeddings/tfidf_embeddings.csv
#   data/embeddings/sbert_embeddings.csv
#   data/embeddings/e5_embeddings.csv
#   data/embeddings/metadata.csv

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

log = logging.getLogger(__name__)

FILINGS_DIR = Path("data/edgar/filings")
EMBED_DIR   = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_DIMS  = 100
SBERT_MODEL = "all-MiniLM-L6-v2"
E5_MODEL    = "intfloat/e5-base-v2"


#~~
# Text cleaning
#~~

def clean_text(text):
    # lowercase and strip numbers and special chars - we want language, not figures
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_texts(filings_dir=FILINGS_DIR):
    # read all txt files and parse ticker/year from filename
    records = []
    for path in sorted(filings_dir.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        ticker = parts[0]
        year   = int(parts[1])
        text   = path.read_text(encoding="utf-8", errors="ignore")
        records.append({"ticker": ticker, "year": year, "text": text, "text_clean": clean_text(text)})

    df = pd.DataFrame(records).sort_values(["ticker", "year"]).reset_index(drop=True)
    log.info(f"Loaded {len(df)} filings ({df['ticker'].nunique()} companies x {sorted(df['year'].unique())} years)")
    return df


#~~
# TF-IDF embeddings
#~~

def run_tfidf(df, n_dims=TFIDF_DIMS, force=False):
    out_path = EMBED_DIR / "tfidf_embeddings.csv"
    if out_path.exists() and not force:
        log.info("Loading TF-IDF embeddings from cache")
        return pd.read_csv(out_path)

    log.info("Building TF-IDF embeddings ...")

    # fit on all documents together so vocabulary is consistent across years
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        min_df=2,           # drop terms appearing in fewer than 2 docs
        max_df=0.90,        # drop terms appearing in more than 90% of docs
        ngram_range=(1, 2)  # include bigrams - "cloud computing" not just "cloud"
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_clean"])
    log.info(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    #-- reduce dimensions via SVD (same idea as latent semantic analysis)
    svd = TruncatedSVD(n_components=n_dims, random_state=42)
    reduced   = svd.fit_transform(tfidf_matrix)
    explained = svd.explained_variance_ratio_.sum()
    log.info(f"  SVD: {tfidf_matrix.shape[1]} -> {n_dims} dims, {explained:.1%} variance kept")

    #-- save with ticker/year so we can join back later
    cols = [f"tfidf_{i}" for i in range(n_dims)]
    out  = pd.DataFrame(reduced, columns=cols)
    out.insert(0, "ticker", df["ticker"].values)
    out.insert(1, "year",   df["year"].values)
    out.to_csv(out_path, index=False)
    log.info(f"  Saved to {out_path}")
    return out


#~~
# Sentence-BERT embeddings
#~~

def run_sbert(df, model_name=SBERT_MODEL, force=False):
    out_path = EMBED_DIR / "sbert_embeddings.csv"
    if out_path.exists() and not force:
        log.info("Loading SBERT embeddings from cache")
        return pd.read_csv(out_path)

    log.info(f"Building SBERT embeddings ({model_name}) ...")
    log.info("  First run downloads the model (~90MB), cached after that.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    model = SentenceTransformer(model_name)

    # 512 token limit per input - take the first ~1500 words which covers the
    # substantive part of Item 1 before the legal boilerplate kicks in
    texts = [" ".join(t.split()[:1500]) for t in df["text_clean"]]

    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    log.info(f"  SBERT shape: {embeddings.shape}")

    cols = [f"sbert_{i}" for i in range(embeddings.shape[1])]
    out  = pd.DataFrame(embeddings, columns=cols)
    out.insert(0, "ticker", df["ticker"].values)
    out.insert(1, "year",   df["year"].values)
    out.to_csv(out_path, index=False)
    log.info(f"  Saved to {out_path}")
    return out


#~~
# E5 embeddings
#~~

def run_e5(df, model_name=E5_MODEL, force=False):
    out_path = EMBED_DIR / "e5_embeddings.csv"
    if out_path.exists() and not force:
        log.info("Loading E5 embeddings from cache")
        return pd.read_csv(out_path)

    log.info(f"Building E5 embeddings ({model_name}) ...")
    log.info("  First run downloads the model (~440MB), cached after that.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    model = SentenceTransformer(model_name)

    # E5 models expect a "passage: " prefix on documents - this is how the model
    # was trained and skipping it noticeably hurts embedding quality
    texts = ["passage: " + " ".join(t.split()[:1500]) for t in df["text_clean"]]

    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    log.info(f"  E5 shape: {embeddings.shape}")

    cols = [f"e5_{i}" for i in range(embeddings.shape[1])]
    out  = pd.DataFrame(embeddings, columns=cols)
    out.insert(0, "ticker", df["ticker"].values)
    out.insert(1, "year",   df["year"].values)
    out.to_csv(out_path, index=False)
    log.info(f"  Saved to {out_path}")
    return out


#~~
# Metadata
#~~

def save_metadata(df):
    meta = df[["ticker", "year"]].copy()
    meta.to_csv(EMBED_DIR / "metadata.csv", index=False)
    return meta


#~~
# Sanity check
#~~

def sanity_check(tfidf_df, sbert_df, e5_df):
    print("\n" + "="*50)
    print("Embeddings check")
    print("="*50)
    for label, emb_df in [("TF-IDF", tfidf_df), ("SBERT", sbert_df), ("E5", e5_df)]:
        prefix = label.lower().replace("-", "")
        # handle tfidf_ prefix
        prefix = "tfidf" if label == "TF-IDF" else label.lower()
        cols   = [c for c in emb_df.columns if c.startswith(prefix)]
        zeros  = (emb_df[cols].abs().sum(axis=1) == 0).sum()
        flag   = "  WARNING: zero rows found" if zeros else ""
        print(f"  {label:<8}  {emb_df.shape[0]} docs x {len(cols)} dims{flag}")
    print(f"\n  Companies: {tfidf_df['ticker'].nunique()}")
    print(f"  Years:     {sorted(tfidf_df['year'].unique())}")
    print("="*50 + "\n")


#~~
# Main
#~~

def run(force=False):
    df       = load_texts()
    save_metadata(df)
    tfidf_df = run_tfidf(df, force=force)
    sbert_df = run_sbert(df, force=force)
    e5_df    = run_e5(df, force=force)
    sanity_check(tfidf_df, sbert_df, e5_df)
    return df, tfidf_df, sbert_df, e5_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    run()
