"""
perfilado.py
------------
Carga las dos fuentes (TMDB e IMDb), produce el perfilado inicial y los
diagnósticos específicos de calidad por fuente.

Salidas:
    outputs/data_quality_report_before.csv
    outputs/diagnostico_tmdb.csv
    outputs/diagnostico_imdb.csv
    figures/profiling_null_pct.png
    figures/profiling_initial_issues.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    IMDB_CANDIDATE_FILES,
    OUT_DIR,
    TMDB_CANDIDATE_FILES,
    ensureDirectories,
)
from utils import (
    findFile,
    normalizeBasicText,
    saveFigure,
    snakeCaseColumns,
    summarizeRows,
    writeCsv,
)


# ============================================================
# Carga
# ============================================================
def loadSources() -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Carga TMDB e IMDb desde disco, normaliza columnas a snake_case y añade _source."""
    tmdbPath = findFile(TMDB_CANDIDATE_FILES)
    imdbPath = findFile(IMDB_CANDIDATE_FILES)

    tmdbRaw = pd.read_csv(tmdbPath, low_memory=False)
    imdbRaw = pd.read_csv(imdbPath, low_memory=False, doublequote=True, escapechar="\\")

    snakeCaseColumns(tmdbRaw)
    snakeCaseColumns(imdbRaw)

    tmdbRaw["_source"] = "tmdb"
    imdbRaw["_source"] = "imdb"

    return tmdbRaw, imdbRaw, tmdbPath, imdbPath


# ============================================================
# Perfilado por columna
# ============================================================
def profileDataframe(df: pd.DataFrame, sourceName: str) -> pd.DataFrame:
    """
    Construye un reporte de calidad por columna: tipo, nulos, unicidad,
    porcentaje sobre filas y muestra de valores.
    """
    rows = []
    nRows = len(df)

    for col in df.columns:
        if col.startswith("_"):
            continue

        s = df[col]
        nNull = int(s.isna().sum())
        nNotNull = int(nRows - nNull)

        try:
            nUnique = int(s.nunique(dropna=True))
        except Exception:
            nUnique = np.nan

        rows.append({
            "source": sourceName,
            "column": col,
            "dtype": str(s.dtype),
            "n_rows": nRows,
            "n_null": nNull,
            "null_pct": round(nNull / nRows * 100, 4) if nRows else np.nan,
            "n_not_null": nNotNull,
            "n_unique": nUnique,
            "unique_pct_over_not_null": (
                round(nUnique / nNotNull * 100, 4) if nNotNull else np.nan
            ),
            "sample_values": str(s.dropna().head(3).tolist()),
        })

    return pd.DataFrame(rows)


# ============================================================
# Diagnósticos específicos
# ============================================================
def diagnoseTmdb(df: pd.DataFrame) -> pd.Series:
    """Diagnostico de problemas conocidos en TMDB."""
    d: Dict[str, int] = {}

    d["n_rows"] = len(df)
    d["n_cols"] = df.shape[1]
    d["duplicated_rows"] = int(df.duplicated().sum())

    if "id" in df.columns:
        d["duplicated_id"] = int(df["id"].duplicated().sum())

    if "title" in df.columns and "release_date" in df.columns:
        tmp = df.copy()
        tmp["title_norm"] = tmp["title"].apply(normalizeBasicText)
        tmp["release_year_tmp"] = pd.to_datetime(
            tmp["release_date"], errors="coerce"
        ).dt.year
        d["duplicated_title_year"] = int(
            tmp.duplicated(subset=["title_norm", "release_year_tmp"]).sum()
        )

    if "runtime" in df.columns:
        rt = pd.to_numeric(df["runtime"], errors="coerce")
        d["runtime_null"] = int(rt.isna().sum())
        d["runtime_zero"] = int((rt == 0).sum())
        d["runtime_gt_600"] = int((rt > 600).sum())

    if "revenue" in df.columns:
        rev = pd.to_numeric(df["revenue"], errors="coerce")
        d["revenue_negative"] = int((rev < 0).sum())
        d["revenue_zero"] = int((rev == 0).sum())

    if "budget" in df.columns:
        bud = pd.to_numeric(df["budget"], errors="coerce")
        d["budget_negative"] = int((bud < 0).sum())
        d["budget_zero"] = int((bud == 0).sum())

    if "vote_average" in df.columns:
        va = pd.to_numeric(df["vote_average"], errors="coerce")
        d["vote_average_zero"] = int((va == 0).sum())
        d["vote_average_out_of_range"] = int(((va < 0) | (va > 10)).sum())

    if "vote_count" in df.columns:
        vc = pd.to_numeric(df["vote_count"], errors="coerce")
        d["vote_count_zero"] = int((vc == 0).sum())

    return pd.Series(d, name="tmdb")


def diagnoseImdb(df: pd.DataFrame) -> pd.Series:
    """Diagnostico de problemas conocidos en IMDb (formato heterogéneo de runtime, año negativo)."""
    d: Dict[str, int] = {}

    d["n_rows"] = len(df)
    d["n_cols"] = df.shape[1]
    d["duplicated_rows"] = int(df.duplicated().sum())

    titleCol = "movie_title" if "movie_title" in df.columns else None
    yearCol = "year" if "year" in df.columns else None

    if titleCol and yearCol:
        tmp = df.copy()
        tmp["title_norm"] = tmp[titleCol].apply(normalizeBasicText)
        tmp["year_num"] = pd.to_numeric(tmp[yearCol], errors="coerce").abs()
        d["duplicated_title_year"] = int(
            tmp.duplicated(subset=["title_norm", "year_num"]).sum()
        )
        # Año con signo negativo (problema sistemático de IMDb)
        d["year_negative"] = int(
            pd.to_numeric(df[yearCol], errors="coerce").lt(0).sum()
        )

    if "run_time" in df.columns:
        rtRaw = df["run_time"].astype(str).str.lower()
        d["run_time_missing"] = int(df["run_time"].isna().sum())
        d["run_time_not_released"] = int(rtRaw.str.contains("not-released", na=False).sum())
        d["run_time_contains_money"] = int(rtRaw.str.contains(r"\$", na=False).sum())

    if "rating" in df.columns:
        rating = pd.to_numeric(df["rating"], errors="coerce")
        d["rating_not_numeric"] = int(rating.isna().sum())
        d["rating_out_of_range"] = int(((rating < 0) | (rating > 10)).sum())

    return pd.Series(d, name="imdb")


# ============================================================
# Gráficas
# ============================================================
def plotNullPercentage(profileDf: pd.DataFrame, sourceName: str) -> Path:
    """Barras horizontales: % de nulos por columna de una fuente."""
    sub = profileDf[profileDf["source"] == sourceName].sort_values("null_pct")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(sub))))
    ax.barh(sub["column"], sub["null_pct"], color="#2E86AB")
    ax.set_xlabel("% de nulos")
    ax.set_title(f"Porcentaje de nulos por columna — {sourceName.upper()}")
    ax.grid(axis="x", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return saveFigure(fig, f"profiling_null_pct_{sourceName}")


def plotInitialIssuesComparison(diagTmdb: pd.Series, diagImdb: pd.Series) -> Path:
    """Comparativa visual TMDB vs IMDb en las dimensiones más críticas."""
    issues = [
        ("títulos\nfaltantes/raros", "duplicated_title_year", "duplicated_title_year"),
        ("años\nfaltantes/inválidos", None, "year_negative"),
        ("runtime\ninválido", "runtime_zero", "run_time_not_released"),
        ("rating\nfaltante/inválido", "vote_count_zero", "rating_not_numeric"),
    ]
    tmdbN = diagTmdb["n_rows"]
    imdbN = diagImdb["n_rows"]

    tmdbPct = [
        100 * diagTmdb.get(t[1], 0) / tmdbN if t[1] else 0 for t in issues
    ]
    imdbPct = [
        100 * diagImdb.get(t[2], 0) / imdbN for t in issues
    ]
    labels = [t[0] for t in issues]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, tmdbPct, w, label=f"TMDB ({tmdbN:,})", color="#2E86AB")
    ax.bar(x + w / 2, imdbPct, w, label=f"IMDb ({imdbN:,})", color="#A23B72")
    for i, (t, im) in enumerate(zip(tmdbPct, imdbPct)):
        ax.text(i - w / 2, t + 1.5, f"{t:.1f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, im + 1.5, f"{im:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% de registros con problema")
    ax.set_title("Perfilado inicial: problemas detectados por fuente")
    ax.legend()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "profiling_initial_issues")


# ============================================================
# Orquestación
# ============================================================
def runProfiling() -> Dict[str, Path]:
    """Ejecuta el bloque completo de perfilado y persiste todos los artefactos."""
    ensureDirectories()

    tmdbRaw, imdbRaw, tmdbPath, imdbPath = loadSources()
    print(f"TMDB: {tmdbPath}")
    print(f"IMDb: {imdbPath}")
    print(summarizeRows(tmdbRaw, "TMDB raw"))
    print(summarizeRows(imdbRaw, "IMDb raw"))

    profileTmdb = profileDataframe(tmdbRaw, "tmdb")
    profileImdb = profileDataframe(imdbRaw, "imdb")
    dataQualityReportBefore = pd.concat([profileTmdb, profileImdb], ignore_index=True)

    diagTmdb = diagnoseTmdb(tmdbRaw)
    diagImdb = diagnoseImdb(imdbRaw)

    paths = {
        "dq_before": writeCsv(
            dataQualityReportBefore, OUT_DIR / "data_quality_report_before.csv"
        ),
        "diag_tmdb": writeCsv(
            diagTmdb.to_frame().reset_index().rename(columns={"index": "metric"}),
            OUT_DIR / "diagnostico_tmdb.csv",
        ),
        "diag_imdb": writeCsv(
            diagImdb.to_frame().reset_index().rename(columns={"index": "metric"}),
            OUT_DIR / "diagnostico_imdb.csv",
        ),
        "fig_null_tmdb": plotNullPercentage(dataQualityReportBefore, "tmdb"),
        "fig_null_imdb": plotNullPercentage(dataQualityReportBefore, "imdb"),
        "fig_issues": plotInitialIssuesComparison(diagTmdb, diagImdb),
    }

    print("\nDiagnóstico TMDB:")
    print(diagTmdb.to_string())
    print("\nDiagnóstico IMDb:")
    print(diagImdb.to_string())

    return paths


if __name__ == "__main__":
    runProfiling()
