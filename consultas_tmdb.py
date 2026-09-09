"""
consultas_tmdb.py
-----------------
Cinco consultas exclusivas sobre el subconjunto TMDB del catálogo maestro:
    Q1. Top 10 películas por revenue observado
    Q2. Problemas financieros y outliers
    Q3. Géneros más frecuentes
    Q4. Popularidad promedio por década
    Q5. Top 25 outliers financieros

Cada consulta produce un CSV en outputs/ y una figura en figures/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUT_DIR, ensureDirectories
from utils import ensureList, plotBar, plotBarh, saveFigure, writeCsv


# ============================================================
# Subconjunto TMDB
# ============================================================
def filterTmdbSubset(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto del catálogo con información TMDB (merged + tmdb_only)."""
    return movieMaster[movieMaster["record_type"].isin(["merged", "tmdb_only"])].copy()


# ============================================================
# Consultas
# ============================================================
def q1TopRevenue(tmdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q1. Top 10 películas TMDB por revenue observado."""
    top = (
        tmdbAnalysis[tmdbAnalysis["revenue_usd"].notna()][
            ["title_display", "release_year", "budget_usd", "revenue_usd",
             "profit_usd", "roi", "rating_normalized", "popularity_tmdb", "record_type"]
        ]
        .sort_values("revenue_usd", ascending=False)
        .head(10)
    )
    writeCsv(top, OUT_DIR / "tmdb_q1_top_10_revenue.csv")
    plotBarh(top.sort_values("revenue_usd"),
             xCol="revenue_usd", yCol="title_display",
             title="TMDB Q1 — Top 10 películas por revenue observado",
             xlabel="Revenue USD", ylabel="Película",
             fname="tmdb_q1_top_revenue")
    return top


def q2FinancialQuality(tmdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q2. Métricas de calidad financiera y outliers."""
    metrics = ["budget_missing", "revenue_missing", "budget_was_zero",
               "revenue_was_zero", "budget_usd_outlier_iqr",
               "revenue_usd_outlier_iqr", "roi_outlier_iqr",
               "multivariate_outlier_iso"]
    counts = {
        "budget_missing": int(tmdbAnalysis["budget_usd"].isna().sum()),
        "revenue_missing": int(tmdbAnalysis["revenue_usd"].isna().sum()),
        "budget_was_zero": int(tmdbAnalysis.get("budget_was_zero", pd.Series(dtype=bool)).sum()),
        "revenue_was_zero": int(tmdbAnalysis.get("revenue_was_zero", pd.Series(dtype=bool)).sum()),
        "budget_usd_outlier_iqr": int(tmdbAnalysis.get("budget_usd_outlier_iqr", pd.Series(dtype=bool)).sum()),
        "revenue_usd_outlier_iqr": int(tmdbAnalysis.get("revenue_usd_outlier_iqr", pd.Series(dtype=bool)).sum()),
        "roi_outlier_iqr": int(tmdbAnalysis.get("roi_outlier_iqr", pd.Series(dtype=bool)).sum()),
        "multivariate_outlier_iso": int(tmdbAnalysis.get("multivariate_outlier_iso", pd.Series(dtype=bool)).sum()),
    }
    out = pd.DataFrame({"metric": metrics, "count": [counts[m] for m in metrics]})
    out["pct"] = (out["count"] / len(tmdbAnalysis) * 100).round(4)
    writeCsv(out, OUT_DIR / "tmdb_q2_financial_quality.csv")
    plotBar(out, xCol="metric", yCol="count",
            title="TMDB Q2 — Problemas financieros y outliers",
            xlabel="Métrica", ylabel="Registros",
            fname="tmdb_q2_financial_quality")
    return out


def q3GenreFrequency(tmdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q3. Top 15 géneros más frecuentes en TMDB."""
    tmdbAnalysis = tmdbAnalysis.copy()
    tmdbAnalysis["genres_list"] = tmdbAnalysis["genres_list"].apply(ensureList)
    exploded = tmdbAnalysis.explode("genres_list")
    freq = (
        exploded["genres_list"]
        .value_counts()
        .head(15)
        .rename_axis("genre")
        .reset_index(name="count")
    )
    freq = freq[freq["genre"] != "Unknown"]
    writeCsv(freq, OUT_DIR / "tmdb_q3_genre_frequency.csv")
    plotBarh(freq.sort_values("count"),
             xCol="count", yCol="genre",
             title="TMDB Q3 — Géneros más frecuentes",
             xlabel="# películas", ylabel="Género",
             fname="tmdb_q3_genre_frequency")
    return freq


def q4PopularityByDecade(tmdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q4. Popularidad promedio por década."""
    sub = tmdbAnalysis[tmdbAnalysis["popularity_tmdb"].notna()
                       & tmdbAnalysis["release_decade"].notna()]
    out = (
        sub.groupby("release_decade")["popularity_tmdb"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "popularity_mean",
                         "median": "popularity_median",
                         "count": "n_movies"})
    )
    out["popularity_mean"] = out["popularity_mean"].round(4)
    out["popularity_median"] = out["popularity_median"].round(4)
    out = out.sort_values("release_decade")
    writeCsv(out, OUT_DIR / "tmdb_q4_popularity_by_decade.csv")
    plotBar(out, xCol="release_decade", yCol="popularity_mean",
            title="TMDB Q4 — Popularidad promedio por década",
            xlabel="Década", ylabel="Popularidad media",
            fname="tmdb_q4_popularity_decade", rotation=0)
    return out


def q5TopOutliers(tmdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q5. Top 25 outliers financieros con todos los flags activos."""
    candidates = tmdbAnalysis[tmdbAnalysis["revenue_usd"].notna()].copy()
    if "budget_usd_outlier_iqr" in candidates.columns:
        candidates["n_flags"] = (
            candidates.get("budget_usd_outlier_iqr", 0).astype(int)
            + candidates.get("revenue_usd_outlier_iqr", 0).astype(int)
            + candidates.get("roi_outlier_iqr", 0).astype(int)
            + candidates.get("multivariate_outlier_iso", 0).astype(int)
        )
    else:
        candidates["n_flags"] = 0
    top = (
        candidates.sort_values(["n_flags", "revenue_usd"], ascending=[False, False])
        .head(25)[["title_display", "release_year", "budget_usd", "revenue_usd",
                   "roi", "rating_normalized", "n_flags"]]
    )
    writeCsv(top, OUT_DIR / "tmdb_q5_outliers_top25.csv")
    plotBarh(top.sort_values("revenue_usd"),
             xCol="revenue_usd", yCol="title_display",
             title="TMDB Q5 — Top 25 outliers financieros (revenue)",
             xlabel="Revenue USD", ylabel="Película",
             fname="tmdb_q5_outliers_top25")
    return top


# ============================================================
# Orquestación
# ============================================================
def runTmdbQueries(movieMaster: pd.DataFrame = None) -> None:
    """Ejecuta las cinco consultas TMDB. Si no se pasa el catálogo, lo carga del disco."""
    ensureDirectories()
    if movieMaster is None:
        movieMaster = pd.read_csv(OUT_DIR / "movie_master.csv", low_memory=False)

    tmdbAnalysis = filterTmdbSubset(movieMaster)
    print(f"TMDB analysis subset: {len(tmdbAnalysis):,} registros")

    q1TopRevenue(tmdbAnalysis)
    q2FinancialQuality(tmdbAnalysis)
    q3GenreFrequency(tmdbAnalysis)
    q4PopularityByDecade(tmdbAnalysis)
    q5TopOutliers(tmdbAnalysis)
    print("Consultas TMDB completadas.")


if __name__ == "__main__":
    runTmdbQueries()
