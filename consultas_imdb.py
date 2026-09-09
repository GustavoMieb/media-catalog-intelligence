"""
consultas_imdb.py
-----------------
Cinco consultas exclusivas sobre el subconjunto IMDb del catálogo maestro:
    Q1. Top 10 películas por rating IMDb
    Q2. Directores más prolíficos
    Q3. Duración promedio por género
    Q4. Campos descriptivos faltantes
    Q5. Rating promedio por década

Cada consulta produce un CSV en outputs/ y una figura en figures/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUT_DIR, ensureDirectories
from utils import ensureList, plotBar, plotBarh, saveFigure, writeCsv


# ============================================================
# Subconjunto IMDb
# ============================================================
def filterImdbSubset(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto con información IMDb (merged + imdb_only)."""
    return movieMaster[movieMaster["record_type"].isin(["merged", "imdb_only"])].copy()


# ============================================================
# Consultas
# ============================================================
def q1TopRating(imdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q1. Top 10 películas por rating IMDb."""
    sub = imdbAnalysis[imdbAnalysis["rating_imdb"].notna()].copy()
    top = (
        sub.sort_values("rating_imdb", ascending=False)
        .head(10)[["title_display", "release_year", "rating_imdb",
                   "user_rating_count_imdb", "director", "record_type"]]
    )
    writeCsv(top, OUT_DIR / "imdb_q1_top_rating.csv")
    plotBarh(top.sort_values("rating_imdb"),
             xCol="rating_imdb", yCol="title_display",
             title="IMDb Q1 — Top 10 películas por rating IMDb",
             xlabel="Rating IMDb", ylabel="Película",
             fname="imdb_q1_top_rating")
    return top


def q2TopDirectors(imdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q2. Directores con mayor número de películas (filtrando ruido)."""
    sub = imdbAnalysis[imdbAnalysis["director"].notna()
                       & (imdbAnalysis["director"].astype(str).str.strip() != "")].copy()
    noise = {"See company contact information", "nan", ""}
    sub = sub[~sub["director"].astype(str).str.strip().isin(noise)]
    top = (
        sub["director"]
        .value_counts()
        .head(15)
        .rename_axis("director")
        .reset_index(name="n_movies")
    )
    writeCsv(top, OUT_DIR / "imdb_q2_top_directors.csv")
    plotBarh(top.sort_values("n_movies"),
             xCol="n_movies", yCol="director",
             title="IMDb Q2 — Directores con más películas",
             xlabel="# películas", ylabel="Director",
             fname="imdb_q2_top_directors")
    return top


def q3RuntimeByGenre(imdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q3. Duración promedio por género (registros con runtime válido)."""
    sub = imdbAnalysis[imdbAnalysis["runtime_min"].notna()].copy()
    sub["genres_list"] = sub["genres_list"].apply(ensureList)
    exploded = sub.explode("genres_list")
    exploded = exploded[exploded["genres_list"] != "Unknown"]
    out = (
        exploded.groupby("genres_list")["runtime_min"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"genres_list": "genre",
                         "mean": "runtime_mean",
                         "median": "runtime_median",
                         "count": "n_movies"})
    )
    out["runtime_mean"] = out["runtime_mean"].round(2)
    out["runtime_median"] = out["runtime_median"].round(2)
    out = out[out["n_movies"] >= 20].sort_values("runtime_mean", ascending=False)
    writeCsv(out, OUT_DIR / "imdb_q3_runtime_by_genre.csv")
    plotBarh(out.sort_values("runtime_mean"),
             xCol="runtime_mean", yCol="genre",
             title="IMDb Q3 — Duración promedio por género",
             xlabel="Runtime promedio (min)", ylabel="Género",
             fname="imdb_q3_runtime_by_genre")
    return out


def q4MissingFields(imdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q4. Porcentaje de nulos en campos editoriales IMDb."""
    fields = ["director", "cast", "overview", "rating_imdb", "runtime_min", "keywords"]
    rows = []
    for f in fields:
        if f not in imdbAnalysis.columns:
            continue
        if imdbAnalysis[f].dtype == object:
            n_null = int((imdbAnalysis[f].isna() | (imdbAnalysis[f].astype(str).str.strip() == "")).sum())
        else:
            n_null = int(imdbAnalysis[f].isna().sum())
        rows.append({
            "field": f,
            "n_null": n_null,
            "pct_null": round(n_null / len(imdbAnalysis) * 100, 4),
        })
    out = pd.DataFrame(rows).sort_values("pct_null", ascending=False)
    writeCsv(out, OUT_DIR / "imdb_q4_missing_fields.csv")
    plotBar(out, xCol="field", yCol="pct_null",
            title="IMDb Q4 — % de nulos en campos descriptivos",
            xlabel="Campo", ylabel="% nulos",
            fname="imdb_q4_missing_fields", rotation=20)
    return out


def q5RatingByDecade(imdbAnalysis: pd.DataFrame) -> pd.DataFrame:
    """Q5. Rating promedio por década."""
    sub = imdbAnalysis[imdbAnalysis["rating_imdb"].notna()
                       & imdbAnalysis["release_decade"].notna()]
    out = (
        sub.groupby("release_decade")["rating_imdb"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "rating_mean",
                         "median": "rating_median",
                         "count": "n_movies"})
    )
    out["rating_mean"] = out["rating_mean"].round(2)
    out = out.sort_values("release_decade")
    writeCsv(out, OUT_DIR / "imdb_q5_rating_by_decade.csv")
    plotBar(out, xCol="release_decade", yCol="rating_mean",
            title="IMDb Q5 — Rating promedio por década",
            xlabel="Década", ylabel="Rating IMDb promedio",
            fname="imdb_q5_rating_by_decade", rotation=0)
    return out


# ============================================================
# Orquestación
# ============================================================
def runImdbQueries(movieMaster: pd.DataFrame = None) -> None:
    """Ejecuta las cinco consultas IMDb."""
    ensureDirectories()
    if movieMaster is None:
        movieMaster = pd.read_csv(OUT_DIR / "movie_master.csv", low_memory=False)

    imdbAnalysis = filterImdbSubset(movieMaster)
    print(f"IMDb analysis subset: {len(imdbAnalysis):,} registros")

    q1TopRating(imdbAnalysis)
    q2TopDirectors(imdbAnalysis)
    q3RuntimeByGenre(imdbAnalysis)
    q4MissingFields(imdbAnalysis)
    q5RatingByDecade(imdbAnalysis)
    print("Consultas IMDb completadas.")


if __name__ == "__main__":
    runImdbQueries()
