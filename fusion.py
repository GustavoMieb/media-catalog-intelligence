"""
fusion.py
---------
Record linkage probabilístico TMDB-IMDb + construcción del Golden Record
(movie_master). Implementa la cascada:
    1) exact title + year
    2) exact title + year +/- 1
    3) fuzzy blocking por (primera letra, año)

Solo los matches con linkage_status = strong_match se fusionan; los
probable_match y manual_review se persisten para auditoría humana.

Salidas:
    outputs/movie_master.csv
    outputs/record_linkage_matches.csv
    outputs/record_linkage_matches_all_review.csv
    outputs/record_linkage_probable_review.csv
    outputs/record_linkage_manual_review.csv
    figures/linkage_similarity_distribution.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from config import (
    OUT_DIR,
    SCORE_PROBABLE_MIN,
    SCORE_STRONG_MIN,
    WEIGHT_GENRE,
    WEIGHT_RUNTIME,
    WEIGHT_TITLE,
    WEIGHT_YEAR,
    ensureDirectories,
)
from limpieza import mainTitleToken, safeFirstLetter
from utils import saveFigure, writeCsv


# ============================================================
# Similaridades parciales
# ============================================================
def runtimeSimilarity(rtA, rtB) -> float:
    """Similitud de duración en 0-100, lineal por tramos."""
    if pd.isna(rtA) or pd.isna(rtB):
        return np.nan
    diff = abs(float(rtA) - float(rtB))
    if diff <= 5:
        return 100.0
    if diff >= 60:
        return 0.0
    return max(0.0, 100.0 * (1.0 - diff / 60.0))


def yearSimilarity(yA, yB) -> float:
    """Función escalonada: 100 si =, 70 si +/-1, 30 si +/-2, 0 otro."""
    if pd.isna(yA) or pd.isna(yB):
        return np.nan
    diff = abs(int(yA) - int(yB))
    if diff == 0:
        return 100.0
    if diff == 1:
        return 70.0
    if diff == 2:
        return 30.0
    return 0.0


def genreSimilarity(genresA, genresB) -> float:
    """Jaccard sobre conjuntos excluyendo 'unknown'."""
    if not isinstance(genresA, list):
        genresA = []
    if not isinstance(genresB, list):
        genresB = []
    a = {str(g).lower().strip() for g in genresA if str(g).strip()}
    b = {str(g).lower().strip() for g in genresB if str(g).strip()}
    a.discard("unknown")
    b.discard("unknown")
    if not a or not b:
        return np.nan
    return 100.0 * len(a & b) / len(a | b)


def compositeLinkageScore(titleSim, yearSim, runtimeSim=np.nan, genreSim=np.nan) -> float:
    """Score compuesto con pesos normalizados según las métricas disponibles."""
    components = []
    if not pd.isna(titleSim):
        components.append((WEIGHT_TITLE, titleSim))
    if not pd.isna(yearSim):
        components.append((WEIGHT_YEAR, yearSim))
    if not pd.isna(runtimeSim):
        components.append((WEIGHT_RUNTIME, runtimeSim))
    if not pd.isna(genreSim):
        components.append((WEIGHT_GENRE, genreSim))
    totalWeight = sum(w for w, _ in components)
    if totalWeight == 0:
        return np.nan
    return sum(w * v for w, v in components) / totalWeight


def classifyMatch(score: float) -> str:
    """Devuelve strong_match | probable_match | manual_review | no_match."""
    if pd.isna(score):
        return "no_match"
    if score >= SCORE_STRONG_MIN:
        return "strong_match"
    if score >= SCORE_PROBABLE_MIN:
        return "probable_match"
    if score >= 75:
        return "manual_review"
    return "no_match"


# ============================================================
# Cascada de matching
# ============================================================
def buildExactMatches(tmdbLink: pd.DataFrame, imdbLink: pd.DataFrame) -> pd.DataFrame:
    """Capa 1: match exacto por (title_norm, release_year)."""
    exact = tmdbLink.merge(
        imdbLink,
        on=["title_norm", "release_year"],
        how="inner",
        suffixes=("_tmdb", "_imdb"),
    )
    matches = pd.DataFrame({
        "tmdb_row_id": exact["tmdb_row_id"],
        "imdb_row_id": exact["imdb_row_id"],
        "tmdb_id": exact["tmdb_id"],
        "imdb_path": exact["imdb_path"],
        "tmdb_title": exact["title_display_tmdb"],
        "imdb_title": exact["title_display_imdb"],
        "tmdb_year": exact["release_year"],
        "imdb_year": exact["release_year"],
        "tmdb_runtime": exact["runtime_min_tmdb"],
        "imdb_runtime": exact["runtime_min_imdb"],
        "title_similarity": 100.0,
        "year_similarity": 100.0,
        "runtime_similarity": [
            runtimeSimilarity(a, b)
            for a, b in zip(exact["runtime_min_tmdb"], exact["runtime_min_imdb"])
        ],
        "genre_similarity": [
            genreSimilarity(a, b)
            for a, b in zip(exact["genres_list_tmdb"], exact["genres_list_imdb"])
        ],
        "linkage_score": 100.0,
        "match_method": "exact_title_year",
        "match_status": "strong_match",
    })
    return resolveOneToOne(matches)


def buildPm1Matches(
    tmdbRemaining: pd.DataFrame, imdbRemaining: pd.DataFrame
) -> pd.DataFrame:
    """Capa 2: título exacto, año +/- 1."""
    titleJoin = tmdbRemaining.merge(
        imdbRemaining, on="title_norm", how="inner", suffixes=("_tmdb", "_imdb")
    )
    titleJoin["year_diff"] = (
        titleJoin["release_year_tmdb"].astype(float)
        - titleJoin["release_year_imdb"].astype(float)
    ).abs()
    pm1 = titleJoin[titleJoin["year_diff"] <= 1].copy()

    rows = []
    for _, r in pm1.iterrows():
        ySim = yearSimilarity(r["release_year_tmdb"], r["release_year_imdb"])
        rtSim = runtimeSimilarity(r["runtime_min_tmdb"], r["runtime_min_imdb"])
        gSim = genreSimilarity(r["genres_list_tmdb"], r["genres_list_imdb"])
        score = compositeLinkageScore(100.0, ySim, rtSim, gSim)
        rows.append({
            "tmdb_row_id": r["tmdb_row_id"],
            "imdb_row_id": r["imdb_row_id"],
            "tmdb_id": r["tmdb_id"],
            "imdb_path": r["imdb_path"],
            "tmdb_title": r["title_display_tmdb"],
            "imdb_title": r["title_display_imdb"],
            "tmdb_year": r["release_year_tmdb"],
            "imdb_year": r["release_year_imdb"],
            "tmdb_runtime": r["runtime_min_tmdb"],
            "imdb_runtime": r["runtime_min_imdb"],
            "title_similarity": 100.0,
            "year_similarity": ySim,
            "runtime_similarity": rtSim,
            "genre_similarity": gSim,
            "linkage_score": score,
            "match_method": "exact_title_year_pm1",
            "match_status": classifyMatch(score),
        })
    return resolveOneToOne(pd.DataFrame(rows))


def buildFuzzyMatches(
    tmdbRemaining: pd.DataFrame, imdbRemaining: pd.DataFrame
) -> pd.DataFrame:
    """Capa 3: fuzzy matching bloqueado por (first_letter, año +/- 1)."""
    # Bloqueo: indexar TMDB restante por (primera letra, año)
    tmdbBlocks: Dict[Tuple[str, int], List[Any]] = {}
    for idx, row in tmdbRemaining.iterrows():
        fl = row["first_letter"]
        yr = row["release_year"]
        if pd.isna(fl) or pd.isna(yr):
            continue
        tmdbBlocks.setdefault((fl, int(yr)), []).append(idx)

    rows = []
    total = len(imdbRemaining)
    for k, (_, rowI) in enumerate(imdbRemaining.iterrows(), start=1):
        if k % 500 == 0:
            print(f"  fuzzy: {k:,}/{total:,}")
        fl = rowI["first_letter"]
        yr = rowI["release_year"]
        if pd.isna(fl) or pd.isna(yr):
            continue

        candidateIdx: List[Any] = []
        for y in [int(yr) - 1, int(yr), int(yr) + 1]:
            candidateIdx.extend(tmdbBlocks.get((fl, y), []))
        if not candidateIdx:
            continue

        candidates = tmdbRemaining.loc[candidateIdx]
        choices = candidates["title_norm"].fillna("").tolist()
        imdbTitle = rowI["title_norm"]
        top = process.extract(imdbTitle, choices, scorer=fuzz.token_set_ratio, limit=5)

        for _matchedTitle, tokenSetScore, pos in top:
            rowT = candidates.iloc[pos]
            tokenSort = fuzz.token_sort_ratio(imdbTitle, rowT["title_norm"])
            simple = fuzz.ratio(imdbTitle, rowT["title_norm"])
            titleSim = max(tokenSetScore, tokenSort, simple)
            ySim = yearSimilarity(rowT["release_year"], rowI["release_year"])
            rtSim = runtimeSimilarity(rowT["runtime_min"], rowI["runtime_min"])
            gSim = genreSimilarity(
                rowT["genres_list_tmdb"], rowI["genres_list_imdb"]
            )
            score = compositeLinkageScore(titleSim, ySim, rtSim, gSim)
            status = classifyMatch(score)
            if status != "no_match":
                rows.append({
                    "tmdb_row_id": rowT["tmdb_row_id"],
                    "imdb_row_id": rowI["imdb_row_id"],
                    "tmdb_id": rowT["tmdb_id"],
                    "imdb_path": rowI["imdb_path"],
                    "tmdb_title": rowT["title_display_tmdb"],
                    "imdb_title": rowI["title_display_imdb"],
                    "tmdb_year": rowT["release_year"],
                    "imdb_year": rowI["release_year"],
                    "tmdb_runtime": rowT["runtime_min"],
                    "imdb_runtime": rowI["runtime_min"],
                    "title_similarity": titleSim,
                    "year_similarity": ySim,
                    "runtime_similarity": rtSim,
                    "genre_similarity": gSim,
                    "linkage_score": score,
                    "match_method": "fuzzy_blocked",
                    "match_status": status,
                })

    return resolveOneToOne(pd.DataFrame(rows))


def resolveOneToOne(matches: pd.DataFrame) -> pd.DataFrame:
    """Greedy 1-a-1: ordena por score desc y descarta duplicados por tmdb y por imdb."""
    if len(matches) == 0:
        return matches
    return (
        matches.sort_values("linkage_score", ascending=False)
        .drop_duplicates(subset=["tmdb_row_id"], keep="first")
        .drop_duplicates(subset=["imdb_row_id"], keep="first")
    )


def runRecordLinkage(
    tmdbClean: pd.DataFrame, imdbClean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta la cascada completa. Devuelve (allMatches, matchesForFusion)
    donde matchesForFusion contiene SOLO los strong_match.
    """
    # Preparar tablas
    tmdbLink = tmdbClean[
        tmdbClean["title_norm"].notna() & tmdbClean["release_year"].notna()
    ].copy()
    imdbLink = imdbClean[
        imdbClean["title_norm"].notna() & imdbClean["release_year"].notna()
    ].copy()
    tmdbLink["first_letter"] = tmdbLink["title_norm"].apply(safeFirstLetter)
    imdbLink["first_letter"] = imdbLink["title_norm"].apply(safeFirstLetter)
    tmdbLink["main_token"] = tmdbLink["title_norm"].apply(mainTitleToken)
    imdbLink["main_token"] = imdbLink["title_norm"].apply(mainTitleToken)

    print(f"Registros TMDB para linkage: {len(tmdbLink):,}")
    print(f"Registros IMDb para linkage: {len(imdbLink):,}")

    # Capa 1: exact
    exact = buildExactMatches(tmdbLink, imdbLink)
    print(f"  exact_title_year: {len(exact):,}")
    matchedTmdb = set(exact["tmdb_row_id"])
    matchedImdb = set(exact["imdb_row_id"])

    # Capa 2: pm1
    tmdbRem = tmdbLink[~tmdbLink["tmdb_row_id"].isin(matchedTmdb)]
    imdbRem = imdbLink[~imdbLink["imdb_row_id"].isin(matchedImdb)]
    pm1 = buildPm1Matches(tmdbRem, imdbRem)
    print(f"  exact_title_year_pm1: {len(pm1):,}")
    matchedTmdb.update(pm1["tmdb_row_id"])
    matchedImdb.update(pm1["imdb_row_id"])

    # Capa 3: fuzzy
    tmdbRem = tmdbLink[~tmdbLink["tmdb_row_id"].isin(matchedTmdb)]
    imdbRem = imdbLink[~imdbLink["imdb_row_id"].isin(matchedImdb)]
    fuzzy = buildFuzzyMatches(tmdbRem, imdbRem)
    print(f"  fuzzy_blocked: {len(fuzzy):,}")

    allMatches = resolveOneToOne(pd.concat([exact, pm1, fuzzy], ignore_index=True))

    # Validaciones de integridad
    assert allMatches["tmdb_row_id"].notna().all()
    assert allMatches["imdb_row_id"].notna().all()
    assert not allMatches["tmdb_row_id"].duplicated().any()
    assert not allMatches["imdb_row_id"].duplicated().any()

    matchesForFusion = allMatches[allMatches["match_status"] == "strong_match"].copy()
    return allMatches, matchesForFusion


# ============================================================
# Detección de conflictos inter-fuente
# ============================================================
def isMissing(x) -> bool:
    """Versión robusta de pd.isna: maneja listas, strings y pd.NA."""
    if x is None:
        return True
    if isinstance(x, list):
        return len(x) == 0 or x == ["Unknown"]
    try:
        if pd.isna(x):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(x, str):
        return x.strip().lower() in ["", "nan", "none", "null", "[]"]
    return False


def safeBool(x, default: bool = False) -> bool:
    try:
        if pd.isna(x):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return bool(x)
    except (TypeError, ValueError):
        return default


def conflictRuntime(a, b, threshold: int = 10) -> bool:
    if isMissing(a) or isMissing(b):
        return False
    return abs(float(a) - float(b)) > threshold


def conflictRating(a, b, threshold: float = 2.0) -> bool:
    if isMissing(a) or isMissing(b):
        return False
    return abs(float(a) - float(b)) > threshold


def conflictYear(a, b, threshold: int = 1) -> bool:
    if isMissing(a) or isMissing(b):
        return False
    return abs(int(a) - int(b)) > threshold


def conflictGenre(a, b) -> bool:
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    sa = {str(x).lower().strip() for x in a if str(x).strip().lower() != "unknown"}
    sb = {str(x).lower().strip() for x in b if str(x).strip().lower() != "unknown"}
    if not sa or not sb:
        return False
    return len(sa & sb) == 0


# ============================================================
# Reglas de supervivencia
# ============================================================
def unionGenres(genresA, genresB) -> List[str]:
    """Unión de géneros conservando orden de primera aparición."""
    result: List[str] = []
    for lista in [genresA, genresB]:
        if isinstance(lista, list):
            for g in lista:
                if pd.isna(g):
                    continue
                gClean = str(g).strip()
                if gClean and gClean.lower() != "unknown" and gClean not in result:
                    result.append(gClean)
    return result if result else ["Unknown"]


def weightedRating(tmdbRating, tmdbCount, imdbRating, imdbCount) -> float:
    """Promedio ponderado de ratings por conteo de votos."""
    ratings = []
    weights = []
    if not isMissing(tmdbRating):
        ratings.append(float(tmdbRating))
        weights.append(float(tmdbCount) if not isMissing(tmdbCount) and float(tmdbCount) > 0 else 1.0)
    if not isMissing(imdbRating):
        ratings.append(float(imdbRating))
        weights.append(float(imdbCount) if not isMissing(imdbCount) and float(imdbCount) > 0 else 1.0)
    if not ratings:
        return np.nan
    return float(np.average(ratings, weights=weights))


def calcProfitRoi(budget, revenue) -> Tuple[float, float, float]:
    """profit, roi, is_profitable solo con datos observados (sin imputación)."""
    if isMissing(budget) or isMissing(revenue):
        return np.nan, np.nan, np.nan
    budget = float(budget)
    revenue = float(revenue)
    if budget <= 0:
        return np.nan, np.nan, np.nan
    profit = revenue - budget
    roi = profit / budget
    return profit, roi, float(revenue > budget)


# ============================================================
# Data Quality Score
# ============================================================
def calcQualityScores(row: pd.Series) -> pd.Series:
    """Calcula los cinco scores dimensionales + DQS compuesto."""
    keyFields = [
        "title_display", "release_year", "runtime_min",
        "genres_canonical", "rating_normalized",
    ]
    financialFields = ["budget_usd", "revenue_usd"]
    allFields = keyFields + financialFields

    present = sum(1 for f in allFields if not isMissing(row.get(f, np.nan)))
    completeness = present / len(allFields)

    validity = float(np.mean([
        not safeBool(row.get("title_missing", False)),
        not safeBool(row.get("year_out_of_range", False)),
        not safeBool(row.get("runtime_out_of_range", False)),
        not safeBool(row.get("rating_missing", False)),
        not safeBool(row.get("genres_missing", False)),
    ]))

    consistency = float(np.mean([
        not safeBool(row.get("runtime_conflict", False)),
        not safeBool(row.get("rating_conflict", False)),
        not safeBool(row.get("genre_conflict", False)),
        not safeBool(row.get("year_conflict", False)),
    ]))

    uniqueness = 1.0  # ya garantizada por dedup + greedy 1-a-1

    sourceCount = row.get("source_count", 1)
    try:
        sourceCount = int(sourceCount) if not pd.isna(sourceCount) else 0
    except (TypeError, ValueError):
        sourceCount = 1
    traceability = 1.0 if sourceCount >= 1 else 0.0

    dqs = (
        0.35 * completeness + 0.25 * validity + 0.20 * consistency
        + 0.10 * uniqueness + 0.10 * traceability
    )

    return pd.Series({
        "completeness_score": round(completeness, 4),
        "validity_score": round(validity, 4),
        "consistency_score": round(consistency, 4),
        "uniqueness_score": round(uniqueness, 4),
        "traceability_score": round(traceability, 4),
        "data_quality_score": round(dqs, 4),
    })


# ============================================================
# Construcción del Golden Record
# ============================================================
def buildMergedRecord(
    match: pd.Series, tmdbRow: pd.Series, imdbRow: pd.Series, idx: int
) -> Dict[str, Any]:
    """Construye un registro merged a partir de un match strong y sus filas fuente."""
    rtConf = conflictRuntime(tmdbRow.get("runtime_min"), imdbRow.get("runtime_min"))
    rtgConf = conflictRating(tmdbRow.get("vote_average_tmdb"), imdbRow.get("rating_imdb"))
    gConf = conflictGenre(tmdbRow.get("genres_list_tmdb"), imdbRow.get("genres_list_imdb"))
    yConf = conflictYear(tmdbRow.get("release_year"), imdbRow.get("release_year"))

    genres = unionGenres(tmdbRow.get("genres_list_tmdb", []), imdbRow.get("genres_list_imdb", []))
    ratingNorm = weightedRating(
        tmdbRow.get("vote_average_tmdb"), tmdbRow.get("vote_count_tmdb"),
        imdbRow.get("rating_imdb"), imdbRow.get("user_rating_count_imdb"),
    )
    profit, roi, isProf = calcProfitRoi(tmdbRow.get("budget_usd"), tmdbRow.get("revenue_usd"))

    titleTmdb = tmdbRow.get("title_display_tmdb")
    titleImdb = imdbRow.get("title_display_imdb")

    return {
        "movie_master_id": f"M_{idx:07d}",
        "record_type": "merged",
        "tmdb_id": tmdbRow.get("tmdb_id"),
        "imdb_path": imdbRow.get("imdb_path"),
        "source_count": 2,
        "source_trace": "tmdb|imdb",
        "title_display": titleTmdb if not isMissing(titleTmdb) else titleImdb,
        "title_display_tmdb": titleTmdb,
        "title_display_imdb": titleImdb,
        "title_norm": tmdbRow.get("title_norm") if not isMissing(tmdbRow.get("title_norm")) else imdbRow.get("title_norm"),
        "release_date": tmdbRow.get("release_date") if not isMissing(tmdbRow.get("release_date")) else imdbRow.get("release_date"),
        "release_year": tmdbRow.get("release_year") if not isMissing(tmdbRow.get("release_year")) else imdbRow.get("release_year"),
        "release_year_tmdb": tmdbRow.get("release_year"),
        "release_year_imdb": imdbRow.get("release_year"),
        "release_decade": tmdbRow.get("release_decade") if not isMissing(tmdbRow.get("release_decade")) else imdbRow.get("release_decade"),
        "year_conflict": yConf,
        "runtime_min": tmdbRow.get("runtime_min") if not isMissing(tmdbRow.get("runtime_min")) else imdbRow.get("runtime_min"),
        "runtime_tmdb": tmdbRow.get("runtime_min"),
        "runtime_imdb": imdbRow.get("runtime_min"),
        "runtime_conflict": rtConf,
        "budget_usd": tmdbRow.get("budget_usd"),
        "revenue_usd": tmdbRow.get("revenue_usd"),
        "budget_observed": safeBool(tmdbRow.get("budget_observed", False)),
        "revenue_observed": safeBool(tmdbRow.get("revenue_observed", False)),
        "profit_usd": profit,
        "roi": roi,
        "is_profitable": isProf,
        "vote_average_tmdb": tmdbRow.get("vote_average_tmdb"),
        "vote_count_tmdb": tmdbRow.get("vote_count_tmdb"),
        "rating_imdb": imdbRow.get("rating_imdb"),
        "user_rating_count_imdb": imdbRow.get("user_rating_count_imdb"),
        "rating_normalized": ratingNorm,
        "rating_conflict": rtgConf,
        "popularity_tmdb": tmdbRow.get("popularity_tmdb"),
        "genres_list": genres,
        "genres_canonical": ", ".join(genres),
        "genre_conflict": gConf,
        "original_language": tmdbRow.get("original_language_clean"),
        "origin_country": tmdbRow.get("origin_country_clean"),
        "director": imdbRow.get("director"),
        "writer": imdbRow.get("writer"),
        "cast": imdbRow.get("cast"),
        "overview": imdbRow.get("overview"),
        "keywords": imdbRow.get("keywords"),
        "production_companies": tmdbRow.get("production_companies"),
        "adult": tmdbRow.get("adult"),
        "linkage_score": match.get("linkage_score"),
        "linkage_status": match.get("match_status"),
        "match_method": match.get("match_method"),
        "title_missing": False,
        "year_out_of_range": safeBool(tmdbRow.get("year_out_of_range", False)) or safeBool(imdbRow.get("year_out_of_range", False)),
        "runtime_out_of_range": safeBool(tmdbRow.get("runtime_out_of_range", False)),
        "rating_missing": isMissing(ratingNorm),
        "genres_missing": genres == ["Unknown"],
    }


def buildTmdbOnlyRecord(t: pd.Series, idx: int) -> Dict[str, Any]:
    """Construye un registro tmdb_only."""
    profit, roi, isProf = calcProfitRoi(t.get("budget_usd"), t.get("revenue_usd"))
    genres = t.get("genres_list_tmdb", ["Unknown"])
    if not isinstance(genres, list) or len(genres) == 0:
        genres = ["Unknown"]
    return {
        "movie_master_id": f"M_{idx:07d}",
        "record_type": "tmdb_only",
        "tmdb_id": t.get("tmdb_id"),
        "imdb_path": np.nan,
        "source_count": 1,
        "source_trace": "tmdb",
        "title_display": t.get("title_display_tmdb"),
        "title_norm": t.get("title_norm"),
        "release_date": t.get("release_date"),
        "release_year": t.get("release_year"),
        "release_decade": t.get("release_decade"),
        "runtime_min": t.get("runtime_min"),
        "budget_usd": t.get("budget_usd"),
        "revenue_usd": t.get("revenue_usd"),
        "budget_observed": safeBool(t.get("budget_observed", False)),
        "revenue_observed": safeBool(t.get("revenue_observed", False)),
        "profit_usd": profit,
        "roi": roi,
        "is_profitable": isProf,
        "vote_average_tmdb": t.get("vote_average_tmdb"),
        "vote_count_tmdb": t.get("vote_count_tmdb"),
        "rating_normalized": t.get("vote_average_tmdb"),
        "popularity_tmdb": t.get("popularity_tmdb"),
        "genres_list": genres,
        "genres_canonical": ", ".join(genres),
        "original_language": t.get("original_language_clean"),
        "origin_country": t.get("origin_country_clean"),
        "production_companies": t.get("production_companies"),
        "adult": t.get("adult"),
        "linkage_score": np.nan,
        "linkage_status": "no_match",
        "match_method": "no_match",
        "title_missing": safeBool(t.get("title_missing", False)),
        "year_out_of_range": safeBool(t.get("year_out_of_range", False)),
        "runtime_out_of_range": safeBool(t.get("runtime_out_of_range", False)),
        "rating_missing": safeBool(t.get("rating_missing", False)),
        "genres_missing": genres == ["Unknown"],
    }


def buildImdbOnlyRecord(i: pd.Series, idx: int) -> Dict[str, Any]:
    """Construye un registro imdb_only."""
    genres = i.get("genres_list_imdb", ["Unknown"])
    if not isinstance(genres, list) or len(genres) == 0:
        genres = ["Unknown"]
    return {
        "movie_master_id": f"M_{idx:07d}",
        "record_type": "imdb_only",
        "tmdb_id": np.nan,
        "imdb_path": i.get("imdb_path"),
        "source_count": 1,
        "source_trace": "imdb",
        "title_display": i.get("title_display_imdb"),
        "title_norm": i.get("title_norm"),
        "release_date": i.get("release_date"),
        "release_year": i.get("release_year"),
        "release_decade": i.get("release_decade"),
        "runtime_min": i.get("runtime_min"),
        "budget_usd": np.nan,
        "revenue_usd": np.nan,
        "budget_observed": False,
        "revenue_observed": False,
        "profit_usd": np.nan,
        "roi": np.nan,
        "is_profitable": np.nan,
        "rating_imdb": i.get("rating_imdb"),
        "user_rating_count_imdb": i.get("user_rating_count_imdb"),
        "rating_normalized": i.get("rating_imdb"),
        "genres_list": genres,
        "genres_canonical": ", ".join(genres),
        "director": i.get("director"),
        "writer": i.get("writer"),
        "cast": i.get("cast"),
        "overview": i.get("overview"),
        "keywords": i.get("keywords"),
        "linkage_score": np.nan,
        "linkage_status": "no_match",
        "match_method": "no_match",
        "title_missing": safeBool(i.get("title_missing", False)),
        "year_out_of_range": safeBool(i.get("year_out_of_range", False)),
        "rating_missing": safeBool(i.get("rating_missing", False)),
        "genres_missing": genres == ["Unknown"],
    }


def buildMovieMaster(
    tmdbClean: pd.DataFrame,
    imdbClean: pd.DataFrame,
    matchesForFusion: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el Golden Record concatenando merged + tmdb_only + imdb_only."""
    tmdbIdx = tmdbClean.set_index("tmdb_row_id", drop=False)
    imdbIdx = imdbClean.set_index("imdb_row_id", drop=False)

    # Merged
    merged: List[Dict[str, Any]] = []
    for _, m in matchesForFusion.iterrows():
        t = tmdbIdx.loc[m["tmdb_row_id"]]
        i = imdbIdx.loc[m["imdb_row_id"]]
        merged.append(buildMergedRecord(m, t, i, len(merged) + 1))
    mergedDf = pd.DataFrame(merged)
    print(f"Merged: {len(mergedDf):,}")

    # tmdb_only
    matchedTmdb = set(matchesForFusion["tmdb_row_id"])
    tmdbOnlySrc = tmdbClean[~tmdbClean["tmdb_row_id"].isin(matchedTmdb)]
    base = len(mergedDf)
    tmdbOnlyRows = [
        buildTmdbOnlyRecord(t, base + k) for k, (_, t) in enumerate(tmdbOnlySrc.iterrows(), start=1)
    ]
    tmdbOnlyDf = pd.DataFrame(tmdbOnlyRows)
    print(f"TMDB only: {len(tmdbOnlyDf):,}")

    # imdb_only
    matchedImdb = set(matchesForFusion["imdb_row_id"])
    imdbOnlySrc = imdbClean[~imdbClean["imdb_row_id"].isin(matchedImdb)]
    base = len(mergedDf) + len(tmdbOnlyDf)
    imdbOnlyRows = [
        buildImdbOnlyRecord(i, base + k) for k, (_, i) in enumerate(imdbOnlySrc.iterrows(), start=1)
    ]
    imdbOnlyDf = pd.DataFrame(imdbOnlyRows)
    print(f"IMDb only: {len(imdbOnlyDf):,}")

    movieMaster = pd.concat([mergedDf, tmdbOnlyDf, imdbOnlyDf], ignore_index=True)

    # Calcular DQS para cada registro
    scores = movieMaster.apply(calcQualityScores, axis=1)
    movieMaster = pd.concat([movieMaster, scores], axis=1)

    # Validación de identidad de consistencia
    expected = len(tmdbClean) + len(imdbClean) - len(matchesForFusion)
    assert len(movieMaster) == expected, (
        f"Identidad de consistencia rota: {len(movieMaster)} != {expected}"
    )

    return movieMaster


# ============================================================
# Gráfica de auditoría
# ============================================================
def plotLinkageScoreDistribution(allMatches: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(allMatches["linkage_score"].dropna(), bins=30, color="#2E86AB", edgecolor="white")
    ax.set_title("Distribución de linkage_score")
    ax.set_xlabel("linkage_score")
    ax.set_ylabel("Frecuencia")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "linkage_similarity_distribution")


# ============================================================
# Orquestación
# ============================================================
def runFusion() -> pd.DataFrame:
    """Ejecuta record linkage + Golden Record y persiste todo."""
    from limpieza import runCleaning  # diferido
    ensureDirectories()

    tmdbClean, imdbClean = runCleaning()

    print("\n=== Record Linkage ===")
    allMatches, matchesForFusion = runRecordLinkage(tmdbClean, imdbClean)
    print(f"\nTotal generados:  {len(allMatches):,}")
    print(f"Strong (fusión):  {len(matchesForFusion):,}")
    print(f"Probable:         {(allMatches['match_status']=='probable_match').sum():,}")
    print(f"Manual review:    {(allMatches['match_status']=='manual_review').sum():,}")

    # Persistencia de auditoría de matches
    writeCsv(allMatches, OUT_DIR / "record_linkage_matches_all_review.csv")
    writeCsv(matchesForFusion, OUT_DIR / "record_linkage_matches.csv")
    writeCsv(
        allMatches[allMatches["match_status"] == "probable_match"],
        OUT_DIR / "record_linkage_probable_review.csv",
    )
    writeCsv(
        allMatches[allMatches["match_status"] == "manual_review"],
        OUT_DIR / "record_linkage_manual_review.csv",
    )

    plotLinkageScoreDistribution(allMatches)

    print("\n=== Golden Record ===")
    movieMaster = buildMovieMaster(tmdbClean, imdbClean, matchesForFusion)
    writeCsv(movieMaster, OUT_DIR / "movie_master.csv")
    print(f"movie_master final: {len(movieMaster):,} registros")
    print(movieMaster["record_type"].value_counts().to_string())

    return movieMaster


if __name__ == "__main__":
    runFusion()
