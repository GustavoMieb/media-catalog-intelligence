"""
limpieza.py
-----------
Pipeline de limpieza intra-fuente. Funciones especializadas por tipo de campo
(texto, fecha, runtime, dinero, conteos, géneros) + ejecución completa que
produce tmdb_clean.csv e imdb_clean.csv.

Filosofía operativa: nunca se imputan budget ni revenue. Los valores cero
en campos financieros se interpretan como faltantes (zero_as_missing).
Los outliers no se eliminan: se marcan con flags.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import GENRE_MAP, OUT_DIR, ensureDirectories
from utils import removeAccents, summarizeRows, writeCsv


# ============================================================
# Normalización de texto
# ============================================================
def normalizeTitle(value: Optional[str]) -> Optional[str]:
    """
    Llave técnica de comparación entre fuentes.
    Minúsculas, sin acentos, sin años entre paréntesis, sin símbolos,
    sin artículo inicial (the/a/an). NO reemplaza el título original.
    """
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    value = removeAccents(value)
    value = re.sub(r"\(\s*\d{4}\s*\)", " ", value)
    value = re.sub(r"\b\d{4}\b", " ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^(the|a|an)\s+", "", value)
    return value if value else np.nan


def parseListField(value: Optional[str]) -> List[str]:
    """Convierte campos serializados como lista a lista Python real."""
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if raw in ["", "[]", "nan", "None", "null"]:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    except Exception:
        parts = re.split(r"[,;|]", raw.strip("[]()"))
        return [p.strip().strip("'").strip('"') for p in parts if p.strip().strip("'").strip('"')]


def normalizeGenreName(g: Optional[str]) -> Optional[str]:
    """Normaliza un género contra el mapeo canónico GENRE_MAP."""
    if pd.isna(g):
        return None
    original = str(g).strip()
    key = removeAccents(original).lower()
    key = re.sub(r"[^a-z0-9\s\-]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    if key in GENRE_MAP:
        return GENRE_MAP[key]
    return original.strip().title()


def normalizeGenres(value: Optional[str]) -> List[str]:
    """Lista normalizada de géneros sin duplicados; ['Unknown'] si vacío."""
    genres = parseListField(value)
    clean: List[str] = []
    for g in genres:
        gNorm = normalizeGenreName(g)
        if gNorm and gNorm not in clean:
            clean.append(gNorm)
    return clean if clean else ["Unknown"]


# ============================================================
# Parsers numéricos especializados
# ============================================================
def parseRuntime(value) -> Tuple[Optional[float], str]:
    """
    Devuelve (minutos, status) con código de estado interpretable.
    Casos especiales conocidos en IMDb: 'not-released', dinero embebido,
    rating embebido (X.X), formato HH:MM, lenguaje natural.
    """
    if pd.isna(value):
        return np.nan, "missing"
    raw = str(value).strip().lower()

    if raw in ["", "nan", "none", "null"]:
        return np.nan, "missing"
    if "not-released" in raw or "not released" in raw:
        return np.nan, "not_released"
    if "$" in raw or "estimated" in raw:
        return np.nan, "money_in_runtime"

    # Formato HH:MM
    m = re.match(r"^(\d{1,3}):(\d{2})$", raw)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2))), "parsed_hhmm"

    # Lenguaje natural: "2 hours 27 minutes"
    m = re.search(
        r"(?:(\d+)\s*(?:hours?|hrs?|hr|h))?\s*(?:(\d+)\s*(?:minutes?|mins?|min|m))?",
        raw,
    )
    if m and (m.group(1) or m.group(2)):
        hours = int(m.group(1)) if m.group(1) else 0
        minutes = int(m.group(2)) if m.group(2) else 0
        total = hours * 60 + minutes
        if total > 0:
            return float(total), "parsed_text"

    # Solo minutos como entero
    m = re.match(r"^(\d+)\s*(?:minutes?|mins?|min|m)?$", raw)
    if m:
        total = float(m.group(1))
        if total < 20:
            return np.nan, "too_small_probably_not_runtime"
        return total, "parsed_minutes"

    # Intento numérico final
    try:
        val = float(raw)
        if val < 20:
            return np.nan, "too_small_probably_rating"
        return val, "parsed_numeric"
    except Exception:
        return np.nan, "unparsed"


def parseMoney(value) -> Tuple[Optional[float], bool, str]:
    """
    Devuelve (valor, was_zero, status).
    Cero se interpreta como faltante (zero_as_missing). No imputa.
    """
    if pd.isna(value):
        return np.nan, False, "missing"
    raw = str(value).strip()
    if raw in ["", "nan", "None", "null"]:
        return np.nan, False, "missing"

    rawClean = re.sub(r"\(.*?\)", "", raw)
    rawClean = rawClean.replace("$", "").replace(",", "").strip()
    try:
        val = float(rawClean)
        if val == 0:
            return np.nan, True, "zero_as_missing"
        if val < 0:
            return np.nan, False, "negative_invalid"
        return val, False, "valid"
    except Exception:
        return np.nan, False, "unparsed"


def parseUserRatingCount(value) -> Optional[float]:
    """Convierte conteos tipo '187K', '1.2M', '500' a número."""
    if pd.isna(value):
        return np.nan
    raw = str(value).strip().upper().replace(",", "")
    if raw in ["", "NAN", "NONE", "NULL"]:
        return np.nan

    m = re.match(r"^([\d\.]+)\s*([KMB]?)$", raw)
    if not m:
        return np.nan
    number = float(m.group(1))
    suffix = m.group(2)
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return number * mult.get(suffix, 1)


def cleanYear(value, validRange: Tuple[int, int] = (1900, 2026)) -> Optional[int]:
    """Limpia un año: maneja string, signo negativo (problema IMDb), y rango válido."""
    try:
        year = abs(int(float(str(value).strip())))
    except (ValueError, TypeError):
        return np.nan
    if year < validRange[0] or year > validRange[1]:
        return np.nan
    return year


# ============================================================
# Helpers para record linkage (usados en fusion.py)
# ============================================================
def safeFirstLetter(titleNorm: Optional[str]) -> Optional[str]:
    """Primera letra del título normalizado (clave de bloqueo)."""
    if pd.isna(titleNorm) or str(titleNorm).strip() == "":
        return np.nan
    return str(titleNorm).strip()[0]


def mainTitleToken(titleNorm: Optional[str]) -> Optional[str]:
    """Primer token del título normalizado."""
    if pd.isna(titleNorm):
        return np.nan
    tokens = str(titleNorm).split()
    return tokens[0] if tokens else np.nan


# ============================================================
# Limpieza por fuente
# ============================================================
def cleanTmdb(tmdbRaw: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo de limpieza a TMDB."""
    df = tmdbRaw.copy()
    df["tmdb_row_id"] = df.index
    df["tmdb_id"] = df["id"].astype(str)

    # Título
    df["title_display_tmdb"] = df["title"].astype(str).str.strip()
    df["title_norm"] = df["title"].apply(normalizeTitle)
    df["title_missing"] = df["title_norm"].isna()

    # Fecha y año
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["release_decade"] = (df["release_year"] // 10 * 10).astype("Int64")
    df["year_missing"] = df["release_year"].isna()
    df["year_out_of_range"] = (df["release_year"] < 1900) | (df["release_year"] > 2026)

    # Runtime
    df["runtime_raw"] = df["runtime"]
    df["runtime_min"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["runtime_was_zero"] = df["runtime_min"].eq(0)
    df["runtime_out_of_range"] = df["runtime_min"].le(0) | df["runtime_min"].gt(600)
    df.loc[df["runtime_out_of_range"], "runtime_min"] = np.nan
    df["runtime_missing"] = df["runtime_min"].isna()

    # Budget
    budgetParsed = df["budget"].apply(parseMoney)
    df["budget_usd"] = budgetParsed.apply(lambda x: x[0])
    df["budget_was_zero"] = budgetParsed.apply(lambda x: x[1])
    df["budget_parse_status"] = budgetParsed.apply(lambda x: x[2])
    df["budget_missing"] = df["budget_usd"].isna()
    df["budget_observed"] = df["budget_usd"].notna()

    # Revenue
    revenueParsed = df["revenue"].apply(parseMoney)
    df["revenue_usd"] = revenueParsed.apply(lambda x: x[0])
    df["revenue_was_zero"] = revenueParsed.apply(lambda x: x[1])
    df["revenue_parse_status"] = revenueParsed.apply(lambda x: x[2])
    df["revenue_missing"] = df["revenue_usd"].isna()
    df["revenue_observed"] = df["revenue_usd"].notna()

    # Ratings TMDB
    df["vote_average_tmdb_raw"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count_tmdb"] = pd.to_numeric(df["vote_count"], errors="coerce")
    df["vote_count_zero"] = df["vote_count_tmdb"].eq(0)
    df["vote_average_zero"] = df["vote_average_tmdb_raw"].eq(0)

    # Si no hay votos, el rating 0 no representa una calificación real
    df["vote_average_tmdb"] = df["vote_average_tmdb_raw"].copy()
    df.loc[
        (df["vote_count_tmdb"].fillna(0) == 0)
        | (df["vote_average_tmdb_raw"] < 0)
        | (df["vote_average_tmdb_raw"] > 10),
        "vote_average_tmdb",
    ] = np.nan
    df["rating_missing"] = df["vote_average_tmdb"].isna()

    # Popularidad
    df["popularity_tmdb"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["popularity_zero"] = df["popularity_tmdb"].eq(0)

    # Géneros
    df["genres_list_tmdb"] = df["genre_names"].apply(normalizeGenres)
    df["genres_missing"] = df["genres_list_tmdb"].apply(lambda x: x == ["Unknown"])

    # Productoras y país
    df["production_companies"] = df["production_company_names"].apply(
        lambda x: ", ".join(parseListField(x))
    )
    df["origin_country_clean"] = df["origin_country"].apply(
        lambda x: ", ".join(parseListField(x))
    )
    df["original_language_clean"] = df["original_language"].astype(str).str.lower().str.strip()

    # Deduplicación por tmdb_id
    nBefore = len(df)
    df = df.sort_values("tmdb_id").drop_duplicates(subset=["tmdb_id"], keep="first")
    print(f"TMDB limpio: {nBefore:,} -> {len(df):,} filas "
          f"({nBefore - len(df):,} duplicados por tmdb_id)")

    return df


def cleanImdb(imdbRaw: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo de limpieza a IMDb."""
    df = imdbRaw.copy()
    df["imdb_row_id"] = df.index
    df["imdb_path"] = df["path"].astype(str).str.strip()

    # Título
    df["title_display_imdb"] = df["movie_title"].astype(str).str.strip()
    df["title_norm"] = df["movie_title"].apply(normalizeTitle)
    df["title_missing"] = df["title_norm"].isna()

    # Año (problema sistemático de signo negativo)
    df["year_raw"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_was_negative"] = df["year_raw"] < 0
    df["release_year"] = df["year_raw"].abs().astype("Int64")
    df["year_missing"] = df["release_year"].isna()
    df["year_out_of_range"] = (df["release_year"] < 1900) | (df["release_year"] > 2026)
    df["release_date"] = pd.to_datetime(
        df["release_year"].astype("string") + "-01-01", errors="coerce"
    )
    df["release_decade"] = (df["release_year"] // 10 * 10).astype("Int64")

    # Runtime con parser robusto
    runtimeParsed = df["run_time"].apply(parseRuntime)
    df["runtime_min"] = runtimeParsed.apply(lambda x: x[0])
    df["runtime_parse_status"] = runtimeParsed.apply(lambda x: x[1])
    df["runtime_missing"] = df["runtime_min"].isna()

    # Rating
    df["rating_imdb"] = pd.to_numeric(df["rating"], errors="coerce")
    df.loc[(df["rating_imdb"] < 0) | (df["rating_imdb"] > 10), "rating_imdb"] = np.nan
    df["rating_missing"] = df["rating_imdb"].isna()

    # Conteo de usuarios
    df["user_rating_count_imdb"] = df["user_rating"].apply(parseUserRatingCount)

    # Géneros
    df["genres_list_imdb"] = df["generes"].apply(normalizeGenres)
    df["genres_missing"] = df["genres_list_imdb"].apply(lambda x: x == ["Unknown"])

    # Texto editorial
    for col in ["overview", "plot_kyeword", "director", "top_5_casts", "writer"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["keywords"] = df["plot_kyeword"]
    df["cast"] = df["top_5_casts"]

    # Deduplicación: conserva el registro de mayor calidad por título+año
    df["dedup_quality_tmp"] = (
        df["rating_imdb"].notna().astype(int)
        + df["runtime_min"].notna().astype(int)
        + df["overview"].ne("").astype(int)
    )
    nBefore = len(df)
    df = (
        df.sort_values(
            ["title_norm", "release_year", "dedup_quality_tmp"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["title_norm", "release_year"], keep="first")
    )
    print(f"IMDb limpio: {nBefore:,} -> {len(df):,} filas "
          f"({nBefore - len(df):,} duplicados por title_norm+release_year)")

    return df


# ============================================================
# Resumen de limpieza
# ============================================================
def buildCleaningSummary(
    tmdbRaw: pd.DataFrame,
    imdbRaw: pd.DataFrame,
    tmdbClean: pd.DataFrame,
    imdbClean: pd.DataFrame,
) -> pd.DataFrame:
    """Tabla comparativa raw vs clean por fuente con principales flags de calidad."""
    return pd.DataFrame({
        "fuente": ["TMDB", "IMDb"],
        "filas_raw": [len(tmdbRaw), len(imdbRaw)],
        "filas_clean": [len(tmdbClean), len(imdbClean)],
        "filas_eliminadas_por_deduplicacion": [
            len(tmdbRaw) - len(tmdbClean),
            len(imdbRaw) - len(imdbClean),
        ],
        "titulos_faltantes": [
            int(tmdbClean["title_missing"].sum()),
            int(imdbClean["title_missing"].sum()),
        ],
        "anios_faltantes": [
            int(tmdbClean["year_missing"].sum()),
            int(imdbClean["year_missing"].sum()),
        ],
        "runtime_faltante_o_invalido": [
            int(tmdbClean["runtime_missing"].sum()),
            int(imdbClean["runtime_missing"].sum()),
        ],
        "rating_faltante_o_invalido": [
            int(tmdbClean["rating_missing"].sum()),
            int(imdbClean["rating_missing"].sum()),
        ],
        "generos_faltantes": [
            int(tmdbClean["genres_missing"].sum()),
            int(imdbClean["genres_missing"].sum()),
        ],
    })


# ============================================================
# Orquestación
# ============================================================
def runCleaning() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga, limpia y persiste ambas fuentes."""
    from perfilado import loadSources  # import diferido: evita ciclos
    ensureDirectories()

    tmdbRaw, imdbRaw, _, _ = loadSources()
    print(summarizeRows(tmdbRaw, "TMDB raw"))
    print(summarizeRows(imdbRaw, "IMDb raw"))

    tmdbClean = cleanTmdb(tmdbRaw)
    imdbClean = cleanImdb(imdbRaw)

    writeCsv(tmdbClean, OUT_DIR / "tmdb_clean.csv")
    writeCsv(imdbClean, OUT_DIR / "imdb_clean.csv")

    summary = buildCleaningSummary(tmdbRaw, imdbRaw, tmdbClean, imdbClean)
    writeCsv(summary, OUT_DIR / "resumen_limpieza.csv")
    print("\nResumen de limpieza:")
    print(summary.to_string(index=False))

    return tmdbClean, imdbClean


if __name__ == "__main__":
    runCleaning()
