# visualize.py
# ThemeDrift
#
# Generates all figures for the report:
#   Fig 1 - UMAP cluster scatter per method x year (3x3 grid)
#   Fig 2 - Company drift tracks: selected companies across 2019→2021→2023
#   Fig 3 - ETF overlap heatmap (F1 scores, cluster x ETF per method)
#   Fig 4 - Temporal lead bar chart per method
#
# Outputs:
#   figures/umap_grid_{constrained|free}.png
#   figures/drift_tracks.png
#   figures/etf_overlap_{method}_{year}.png
#   figures/temporal_lead.png
#
# Install before running:
#   pip install matplotlib seaborn

import logging
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

log = logging.getLogger(__name__)

CLUSTER_DIR    = Path("data/clusters/constrained")
VALIDATION_DIR = Path("data/validation")
FIGURES_DIR    = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["tfidf", "sbert", "e5"]
YEARS   = [2019, 2021, 2023]

METHOD_LABELS = {"tfidf": "TF-IDF", "sbert": "SBERT", "e5": "E5"}

CLUSTER_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

NOISE_COLOR = "#bbbbbb"

DRIFT_HIGHLIGHT = ["NVDA", "MSFT", "TSLA", "MA", "V", "AMZN", "CRM", "AMD"]

# ── light theme ──────────────────────────────────────────────────
BG       = "white"
PANEL_BG = "#f7f7f7"
SPINE    = "#cccccc"
TICK     = "#555555"
TEXT     = "#222222"
DIM      = "#888888"


#~~
# Helpers
#~~

def load_umap(method, year, constrained=True):
    label = "constrained" if constrained else "free"
    path  = Path("data/clusters") / label / f"umap_{method}_{year}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_assignments(method, year, constrained=True):
    label = "constrained" if constrained else "free"
    path  = Path("data/clusters") / label / f"assignments_{method}_{year}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def cluster_color(cluster_id):
    if cluster_id == -1:
        return NOISE_COLOR
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


#~~
# Fig 1 - UMAP grid (3 methods x 3 years)
#~~

def plot_umap_grid(constrained=True, force=False):
    label    = "constrained" if constrained else "free"
    out_path = FIGURES_DIR / f"umap_grid_{label}.png"
    if out_path.exists() and not force:
        log.info(f"  Skipping UMAP grid (cached): {out_path}")
        return

    fig, axes = plt.subplots(len(YEARS), len(METHODS), figsize=(18, 15), facecolor=BG)
    fig.suptitle(f"ThemeDrift — UMAP Cluster Layout  [{label}]",
                 fontsize=16, color=TEXT, y=0.98, fontweight="bold")

    for row, year in enumerate(YEARS):
        for col, method in enumerate(METHODS):
            ax = axes[row][col]
            ax.set_facecolor(PANEL_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE)

            umap_df = load_umap(method, year, constrained)
            if umap_df is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        color=DIM, transform=ax.transAxes)
                continue

            for _, r in umap_df.iterrows():
                cid   = int(r["cluster"])
                color = cluster_color(cid)
                alpha = 0.4 if cid == -1 else 0.85
                size  = 35  if cid == -1 else 60
                ax.scatter(r["x"], r["y"], c=color, alpha=alpha, s=size, zorder=3)

            for _, r in umap_df.iterrows():
                if int(r["cluster"]) == -1:
                    continue
                ax.annotate(r["ticker"], (r["x"], r["y"]),
                            fontsize=5.5, color="#333333", alpha=0.9,
                            xytext=(3, 3), textcoords="offset points", zorder=4)

            n_themes = umap_df[umap_df["cluster"] != -1]["cluster"].nunique()
            n_noise  = (umap_df["cluster"] == -1).sum()

            if row == 0:
                ax.set_title(METHOD_LABELS[method], color=TEXT,
                             fontsize=13, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(str(year), color=TICK, fontsize=11, labelpad=8)

            ax.tick_params(colors=TICK, labelsize=7)
            ax.set_xlabel(f"{n_themes} themes · {n_noise} noise", color=DIM, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


#~~
# Fig 2 - Drift tracks
#~~

def plot_drift_tracks(method="tfidf", constrained=True, force=False):
    out_path = FIGURES_DIR / "drift_tracks.png"
    if out_path.exists() and not force:
        log.info(f"  Skipping drift tracks (cached): {out_path}")
        return

    year_data = {}
    for year in YEARS:
        df = load_umap(method, year, constrained)
        if df is not None:
            year_data[year] = df

    if len(year_data) < 2:
        log.warning("  Not enough year data for drift tracks")
        return

    fig, axes = plt.subplots(1, len(YEARS), figsize=(18, 6), facecolor=BG)
    fig.suptitle(f"Company Theme Drift  ({METHOD_LABELS[method]})  2019 → 2021 → 2023",
                 fontsize=14, color=TEXT, y=1.01, fontweight="bold")

    highlight_colors = {t: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                        for i, t in enumerate(DRIFT_HIGHLIGHT)}

    for col, year in enumerate(YEARS):
        ax = axes[col]
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

        df = year_data.get(year)
        if df is None:
            continue

        noise_mask = df["cluster"] == -1
        ax.scatter(df.loc[noise_mask,  "x"], df.loc[noise_mask,  "y"],
                   c=NOISE_COLOR, s=18, alpha=0.4, zorder=2)
        ax.scatter(df.loc[~noise_mask, "x"], df.loc[~noise_mask, "y"],
                   c="#cccccc", s=30, alpha=0.6, zorder=2)

        for ticker in DRIFT_HIGHLIGHT:
            r = df[df["ticker"] == ticker]
            if r.empty:
                continue
            x, y  = r["x"].values[0], r["y"].values[0]
            color = highlight_colors[ticker]
            cid   = int(r["cluster"].values[0])
            ax.scatter(x, y, c=color, s=120, zorder=5,
                       edgecolors="#333333", linewidths=0.8,
                       marker="D" if cid == -1 else "o")
            ax.annotate(ticker, (x, y), fontsize=7.5, color=color, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=6)

        ax.set_title(str(year), color=TICK, fontsize=12, pad=8)
        ax.tick_params(colors=TICK, labelsize=6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=highlight_colors[t], markeredgecolor="#333333",
               markeredgewidth=0.5, markersize=8, label=t, linestyle="None")
        for t in DRIFT_HIGHLIGHT
    ]
    legend_handles.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor=NOISE_COLOR,
               markeredgecolor="#333333", markeredgewidth=0.5,
               markersize=7, label="noise (diamond)", linestyle="None")
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(DRIFT_HIGHLIGHT) + 1,
               frameon=False, fontsize=8, labelcolor=TEXT, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


#~~
# Fig 3 - ETF overlap heatmap
#~~

def plot_etf_heatmap(method, year, force=False):
    out_path = FIGURES_DIR / f"etf_overlap_{method}_{year}.png"
    if out_path.exists() and not force:
        log.info(f"  Skipping ETF heatmap (cached): {out_path}")
        return

    path = VALIDATION_DIR / f"overlap_{method}_{year}.csv"
    if not path.exists():
        log.warning(f"  No overlap data: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    pivot = df.pivot_table(index="cluster_id", columns="etf", values="f1", aggfunc="max").fillna(0)
    pivot.index   = [f"C{int(i)}" for i in pivot.index]
    pivot.columns = [c.split("_")[0] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1),
                                    max(5, len(pivot) * 0.55)),
                           facecolor=BG)
    ax.set_facecolor(BG)

    try:
        import seaborn as sns
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                    cmap="YlOrRd", vmin=0, vmax=1,
                    linewidths=0.4, linecolor="#eeeeee",
                    annot_kws={"size": 8, "color": "#222222"},
                    cbar_kws={"shrink": 0.7})
    except ImportError:
        im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
        plt.colorbar(im, ax=ax, shrink=0.7)

    ax.set_title(f"ETF Overlap F1  ·  {METHOD_LABELS[method]}  {year}",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Thematic ETF", color=TICK, fontsize=9)
    ax.set_ylabel("Cluster",      color=TICK, fontsize=9)
    ax.tick_params(colors=TICK, labelsize=8)

    try:
        ax.collections[-1].colorbar.ax.tick_params(colors=TICK, labelsize=7)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


#~~
# Fig 4 - Temporal lead
#~~

def plot_temporal_lead(force=False):
    out_path = FIGURES_DIR / "temporal_lead.png"
    if out_path.exists() and not force:
        log.info(f"  Skipping temporal lead chart (cached): {out_path}")
        return

    rows = []
    for method in METHODS:
        path = VALIDATION_DIR / f"lead_time_{method}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        detected = df[df["detected_early"] == True]
        if detected.empty:
            continue
        for _, r in detected.iterrows():
            rows.append({"method": METHOD_LABELS[method], "company": r["company"],
                         "etf": r["etf"].split("_")[0], "lead_years": int(r["lead_years"])})

    if not rows:
        log.warning("  No temporal lead data found - skipping")
        return

    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)

    # ── left: avg lead bar ───────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    for spine in ax1.spines.values():
        spine.set_edgecolor(SPINE)

    method_avg = (data.groupby("method")["lead_years"]
                  .agg(["mean", "count"]).reset_index()
                  .rename(columns={"mean": "avg_lead", "count": "n_detected"}))

    bars = ax1.bar(method_avg["method"], method_avg["avg_lead"],
                   color=[CLUSTER_PALETTE[i] for i in range(len(method_avg))],
                   width=0.5, alpha=0.85, zorder=3)

    for bar, (_, mrow) in zip(bars, method_avg.iterrows()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{mrow['avg_lead']:.1f}y\n(n={int(mrow['n_detected'])})",
                 ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

    ax1.set_title("Avg Temporal Lead by Method", color=TEXT, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Years ahead of ETF inclusion", color=TICK, fontsize=9)
    ax1.tick_params(colors=TICK, labelsize=9)
    ax1.set_ylim(0, max(method_avg["avg_lead"]) * 1.4)
    ax1.axhline(2, color=SPINE, linestyle="--", linewidth=0.8, zorder=2)
    ax1.text(ax1.get_xlim()[1] * 0.98, 2.05, "2yr", color=DIM, fontsize=7, ha="right")

    # ── right: bubble scatter ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    for spine in ax2.spines.values():
        spine.set_edgecolor(SPINE)

    sub = data[data["method"] == "TF-IDF"].copy()
    if sub.empty:
        sub = data.copy()

    etfs      = sorted(sub["etf"].unique())
    companies = sorted(sub["company"].unique())
    etf_idx   = {e: i for i, e in enumerate(etfs)}
    comp_idx  = {c: i for i, c in enumerate(companies)}

    import random
    rng = random.Random(42)  # fixed seed so jitter is reproducible

    for _, r in sub.iterrows():
        lead  = r["lead_years"]
        color = "#59a14f" if lead >= 2 else "#f28e2b"
        # compute jitter once, reuse for both scatter and annotate
        jx = rng.uniform(-0.12, 0.12)
        jy = rng.uniform(-0.12, 0.12)
        x  = etf_idx[r["etf"]]  + jx
        y  = comp_idx[r["company"]] + jy
        ax2.scatter(x, y, s=lead * 100, c=color, alpha=0.75, zorder=3,
                    edgecolors="#333333", linewidths=0.5)
        ax2.annotate(f"{lead}y", (x, y),
                     ha="center", va="center", fontsize=6.5,
                     color="white", fontweight="bold", zorder=4)

    ax2.set_xticks(range(len(etfs)))
    ax2.set_xticklabels(etfs, rotation=40, ha="right", color=TICK, fontsize=8)
    ax2.set_yticks(range(len(companies)))
    ax2.set_yticklabels(companies, color=TICK, fontsize=8)
    ax2.set_title("Lead Time by Company x ETF  (TF-IDF)", color=TEXT,
                  fontsize=12, fontweight="bold")
    ax2.legend(handles=[mpatches.Patch(color="#59a14f", label="≥2yr lead"),
                        mpatches.Patch(color="#f28e2b", label="1yr lead")],
               frameon=True, fontsize=8, labelcolor=TEXT,
               loc="lower right", edgecolor=SPINE, facecolor=BG)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


#~~
# Run all figures
#~~

def run(constrained=True, force=False):
    log.info("Generating figures ...")
    plot_umap_grid(constrained=constrained, force=force)
    plot_drift_tracks(method="tfidf", constrained=constrained, force=force)
    for method in METHODS:
        for year in YEARS:
            plot_etf_heatmap(method, year, force=force)
    plot_temporal_lead(force=force)
    log.info(f"  All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    run(constrained=True, force=False)
