"""
consultas_multifuente.py
------------------------
Diez consultas integradas que combinan campos de TMDB e IMDb sobre los
registros merged del catálogo maestro. Cada una sería imposible de
contestar usando una sola fuente.

    Q1.  Top matches por linkage_score
    Q2.  Películas merged con títulos distintos entre TMDB e IMDb
    Q3.  Mayor desacuerdo entre rating TMDB e IMDb
    Q4.  Mayor conflicto de runtime
    Q5.  Alto rating IMDb con bajo revenue
    Q6.  Alto revenue con bajo rating IMDb
    Q7.  Directores con mayor revenue promedio
    Q8.  ROI mediano por género
    Q9.  Rentabilidad por década
    Q10. Alto potencial comercial sin conflictos inter-fuente

Cada consulta produce un CSV en outputs/ y una figura en figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT_DIR, ensureDirectories
from utils import ensureList, plotBar, plotBarh, saveFigure, writeCsv


# ============================================================
# Subconjunto integrado
# ============================================================
def filterMergedSubset(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Solo registros merged: aquí coexisten ambas fuentes."""
    return movieMaster[movieMaster["record_type"] == "merged"].copy()


# ============================================================
# Consultas
# ============================================================
def q1TopLinkage(merged: pd.DataFrame) -> pd.DataFrame:
    """Q1. Top 20 matches por linkage_score con datos cruzados."""
    cols = ["title_display", "release_year", "linkage_score", "match_method",
            "vote_average_tmdb", "rating_imdb", "rating_normalized"]
    top = (
        merged[merged["linkage_score"].notna()]
        .sort_values("linkage_score", ascending=False)
        .head(20)[cols]
    )
    writeCsv(top, OUT_DIR / "integrated_q1_top_linkage.csv")

    plotBarh(top.sort_values("linkage_score"),
             xCol="linkage_score", yCol="title_display",
             title="INT Q1 — Top matches por linkage_score",
             xlabel="Linkage score", ylabel="Película",
             fname="int_q1_top_linkage")
    return top


def q2TitleDifferences(merged: pd.DataFrame) -> pd.DataFrame:
    """Q2. Películas merged con títulos distintos entre fuentes."""
    sub = merged[
        merged["title_display_tmdb"].notna()
        & merged["title_display_imdb"].notna()
    ].copy()
    sub["different"] = (
        sub["title_display_tmdb"].astype(str).str.strip().str.lower()
        != sub["title_display_imdb"].astype(str).str.strip().str.lower()
    )
    diff = sub[sub["different"]][[
        "title_display_tmdb", "title_display_imdb",
        "release_year", "linkage_score", "match_method"
    ]].head(30)
    writeCsv(diff, OUT_DIR / "integrated_q2_title_differences.csv")

    # Visualizar: # de pares con título distinto por match_method
    by_method = (
        sub[sub["different"]]["match_method"]
        .value_counts()
        .rename_axis("match_method")
        .reset_index(name="count")
    )
    plotBar(by_method, xCol="match_method", yCol="count",
            title="INT Q2 — Títulos distintos por método de match",
            xlabel="Match method", ylabel="# pares",
            fname="int_q2_title_differences", rotation=0)
    return diff


def q3RatingDisagreement(merged: pd.DataFrame) -> pd.DataFrame:
    """Q3. Películas con mayor desacuerdo entre rating TMDB e IMDb."""
    sub = merged[
        merged["vote_average_tmdb"].notna() & merged["rating_imdb"].notna()
    ].copy()
    sub["rating_diff"] = (sub["vote_average_tmdb"] - sub["rating_imdb"]).abs()
    top = sub.sort_values("rating_diff", ascending=False).head(25)[[
        "title_display", "release_year", "vote_average_tmdb",
        "vote_count_tmdb", "rating_imdb", "user_rating_count_imdb",
        "rating_diff"
    ]]
    writeCsv(top, OUT_DIR / "integrated_q3_rating_disagreement.csv")

    plotBarh(top.sort_values("rating_diff"),
             xCol="rating_diff", yCol="title_display",
             title="INT Q3 — Mayor diferencia entre rating TMDB e IMDb",
             xlabel="|Δ rating|", ylabel="Película",
             fname="int_q3_rating_disagreement")
    return top


def q4RuntimeConflicts(merged: pd.DataFrame) -> pd.DataFrame:
    """Q4. Mayor diferencia de runtime entre fuentes."""
    sub = merged[
        merged["runtime_tmdb"].notna() & merged["runtime_imdb"].notna()
    ].copy()
    sub["runtime_diff"] = (sub["runtime_tmdb"] - sub["runtime_imdb"]).abs()
    top = sub.sort_values("runtime_diff", ascending=False).head(25)[[
        "title_display", "release_year", "runtime_tmdb",
        "runtime_imdb", "runtime_diff"
    ]]
    writeCsv(top, OUT_DIR / "integrated_q4_runtime_conflicts.csv")

    plotBarh(top.sort_values("runtime_diff"),
             xCol="runtime_diff", yCol="title_display",
             title="INT Q4 — Mayor diferencia de runtime entre TMDB e IMDb",
             xlabel="|Δ runtime| (min)", ylabel="Película",
             fname="int_q4_runtime_conflicts")
    return top


def q5HighImdbLowRevenue(merged: pd.DataFrame) -> pd.DataFrame:
    """Q5. Películas con rating IMDb muy alto pero bajo revenue."""
    sub = merged[
        (merged["rating_imdb"] >= 8.4)
        & merged["revenue_usd"].notna()
        & (merged["revenue_usd"] < 2_500_000)
    ].copy()
    top = sub.sort_values("rating_imdb", ascending=False).head(25)[[
        "title_display", "release_year", "rating_imdb",
        "revenue_usd", "budget_usd", "roi", "director"
    ]]
    writeCsv(top, OUT_DIR / "integrated_q5_high_rating_low_revenue.csv")

    plotBarh(top.sort_values("rating_imdb"),
             xCol="rating_imdb", yCol="title_display",
             title="INT Q5 — Alto rating IMDb con bajo revenue",
             xlabel="Rating IMDb", ylabel="Película",
             fname="int_q5_high_rating_low_revenue")
    return top


def q6HighRevenueLowRating(merged: pd.DataFrame) -> pd.DataFrame:
    """Q6. Películas con revenue alto pero rating IMDb bajo."""
    sub = merged[
        (merged["rating_imdb"] < 5.5)
        & (merged["revenue_usd"] > 200_000_000)
    ].copy()
    top = sub.sort_values("revenue_usd", ascending=False).head(25)[[
        "title_display", "release_year", "revenue_usd",
        "rating_imdb", "vote_average_tmdb", "roi"
    ]]
    writeCsv(top, OUT_DIR / "integrated_q6_high_revenue_low_rating.csv")

    plotBarh(top.sort_values("revenue_usd"),
             xCol="revenue_usd", yCol="title_display",
             title="INT Q6 — Alto revenue con bajo rating IMDb",
             xlabel="Revenue USD", ylabel="Película",
             fname="int_q6_high_revenue_low_rating")
    return top


def q7DirectorRevenue(merged: pd.DataFrame) -> pd.DataFrame:
    """Q7. Directores con mayor revenue promedio (al menos 3 películas)."""
    sub = merged[
        merged["director"].notna()
        & (merged["director"].astype(str).str.strip() != "")
        & merged["revenue_usd"].notna()
    ].copy()
    noise = {"See company contact information", "nan", ""}
    sub = sub[~sub["director"].astype(str).str.strip().isin(noise)]

    out = (
        sub.groupby("director")
        .agg(n_movies=("revenue_usd", "count"),
             revenue_mean=("revenue_usd", "mean"),
             revenue_total=("revenue_usd", "sum"),
             roi_mean=("roi", "mean"),
             rating_mean=("rating_imdb", "mean"))
        .reset_index()
    )
    out = out[out["n_movies"] >= 3].copy()
    out["revenue_mean"] = out["revenue_mean"].round(0)
    out["roi_mean"] = out["roi_mean"].round(2)
    out["rating_mean"] = out["rating_mean"].round(2)
    out = out.sort_values("revenue_mean", ascending=False).head(20)
    writeCsv(out, OUT_DIR / "integrated_q7_director_revenue.csv")

    plotBarh(out.sort_values("revenue_mean"),
             xCol="revenue_mean", yCol="director",
             title="INT Q7 — Directores con mayor revenue promedio",
             xlabel="Revenue promedio (USD)", ylabel="Director",
             fname="int_q7_director_revenue")
    return out


def q8GenreRoi(merged: pd.DataFrame) -> pd.DataFrame:
    """Q8. ROI mediano por género."""
    sub = merged[merged["roi"].notna()].copy()
    sub["genres_list"] = sub["genres_list"].apply(ensureList)
    exploded = sub.explode("genres_list")
    exploded = exploded[exploded["genres_list"] != "Unknown"]
    out = (
        exploded.groupby("genres_list")["roi"]
        .agg(["median", "mean", "count"])
        .reset_index()
        .rename(columns={"genres_list": "genre",
                         "median": "roi_median",
                         "mean": "roi_mean",
                         "count": "n_movies"})
    )
    out["roi_median"] = out["roi_median"].round(2)
    out["roi_mean"] = out["roi_mean"].round(2)
    out = out[out["n_movies"] >= 20].sort_values("roi_median", ascending=False)
    writeCsv(out, OUT_DIR / "integrated_q8_genre_roi.csv")

    plotBarh(out.sort_values("roi_median"),
             xCol="roi_median", yCol="genre",
             title="INT Q8 — ROI mediano por género",
             xlabel="ROI mediano", ylabel="Género",
             fname="int_q8_genre_roi")
    return out


def q9DecadeProfitability(merged: pd.DataFrame) -> pd.DataFrame:
    """Q9. Porcentaje de películas rentables por década."""
    sub = merged[
        merged["is_profitable"].notna()
        & merged["release_decade"].notna()
    ].copy()
    out = (
        sub.groupby("release_decade")
        .agg(profitable_pct=("is_profitable", "mean"),
             n_movies=("is_profitable", "count"))
        .reset_index()
    )
    out["profitable_pct"] = (out["profitable_pct"] * 100).round(2)
    out = out.sort_values("release_decade")
    writeCsv(out, OUT_DIR / "integrated_q9_decade_profitability.csv")

    plotBar(out, xCol="release_decade", yCol="profitable_pct",
            title="INT Q9 — % de películas rentables por década",
            xlabel="Década", ylabel="% rentables",
            fname="int_q9_decade_profitability", rotation=0)
    return out


def q10HighPotentialNoConflicts(merged: pd.DataFrame) -> pd.DataFrame:
    """Q10. Películas de alto potencial comercial sin conflictos inter-fuente."""
    confCols = ["runtime_conflict", "rating_conflict",
                "year_conflict", "genre_conflict"]
    sub = merged.copy()
    if all(c in sub.columns for c in confCols):
        sub = sub[~sub[confCols].any(axis=1)]
    sub = sub[
        merged["revenue_usd"].notna()
        & merged["rating_imdb"].notna()
        & (merged["rating_imdb"] >= 7.5)
    ]
    sub["commercial_potential_score"] = (
        np.log1p(sub["revenue_usd"]) * sub["rating_imdb"] / 10.0
    )
    top = (
        sub.sort_values("commercial_potential_score", ascending=False)
        .head(25)[["title_display", "release_year", "revenue_usd",
                   "rating_imdb", "rating_normalized",
                   "commercial_potential_score"]]
    )
    writeCsv(top, OUT_DIR / "integrated_q10_low_quality_high_potential.csv")

    plotBarh(top.sort_values("commercial_potential_score"),
             xCol="commercial_potential_score", yCol="title_display",
             title="INT Q10 — Alto potencial comercial sin conflictos",
             xlabel="Commercial potential score", ylabel="Película",
             fname="int_q10_low_quality_high_potential")
    return top


# ============================================================
# Orquestación
# ============================================================
def runIntegratedQueries(movieMaster: pd.DataFrame = None) -> None:
    """Ejecuta las diez consultas integradas."""
    ensureDirectories()
    if movieMaster is None:
        movieMaster = pd.read_csv(OUT_DIR / "movie_master.csv", low_memory=False)

    merged = filterMergedSubset(movieMaster)
    print(f"Registros merged: {len(merged):,}")

    q1TopLinkage(merged)
    q2TitleDifferences(merged)
    q3RatingDisagreement(merged)
    q4RuntimeConflicts(merged)
    q5HighImdbLowRevenue(merged)
    q6HighRevenueLowRating(merged)
    q7DirectorRevenue(merged)
    q8GenreRoi(merged)
    q9DecadeProfitability(merged)
    q10HighPotentialNoConflicts(merged)
    print("Consultas integradas completadas.")


if __name__ == "__main__":
    runIntegratedQueries()
