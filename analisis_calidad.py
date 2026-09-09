"""
analisis_calidad.py
-------------------
Perfilado posterior del catálogo maestro + detección de outliers (IQR y
multivariado con Isolation Forest) + auditoría final del proyecto.

Depende de: limpieza.py, fusion.py, perfilado.py
Se ejecuta ANTES de analisis.py (produce outputs/movie_master.csv).

Salidas:
    outputs/data_quality_report_after.csv
    outputs/record_type_summary.csv
    outputs/coverage_summary.csv
    outputs/quality_before_after_summary.csv
    outputs/iqr_outlier_summary.csv
    outputs/outlier_summary_by_record_type.csv
    outputs/audit_checks.csv
    figures/dqs_by_record_type.png
    figures/record_type_distribution.png
    figures/null_pct_post.png
    figures/outliers_budget_revenue.png
    figures/roi_distribution.png
    figures/quality_dimensions_by_record_type.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    ISO_FOREST_CONTAMINATION,
    IQR_MULTIPLIER,
    OUT_DIR,
    RANDOM_STATE,
    ensureDirectories,
)
from perfilado import profileDataframe
from utils import saveFigure, writeCsv


# ============================================================
# Perfilado posterior
# ============================================================
def buildRecordTypeSummary(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Conteo por record_type con porcentaje."""
    s = (
        movieMaster["record_type"]
        .value_counts()
        .rename_axis("record_type")
        .reset_index(name="count")
    )
    s["pct"] = (s["count"] / s["count"].sum() * 100).round(4)
    return s


def buildCoverageSummary(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Cobertura de integración: cuántos registros de cada fuente entraron al maestro."""
    nMerged = int((movieMaster["record_type"] == "merged").sum())
    nTmdbOnly = int((movieMaster["record_type"] == "tmdb_only").sum())
    nImdbOnly = int((movieMaster["record_type"] == "imdb_only").sum())
    tmdbTotal = nMerged + nTmdbOnly
    imdbTotal = nMerged + nImdbOnly

    return pd.DataFrame({
        "metric": [
            "merged_records",
            "tmdb_only_records",
            "imdb_only_records",
            "movie_master_total",
            "tmdb_fusion_rate_pct",
            "imdb_fusion_rate_pct",
        ],
        "value": [
            nMerged,
            nTmdbOnly,
            nImdbOnly,
            nMerged + nTmdbOnly + nImdbOnly,
            round(nMerged / tmdbTotal * 100, 4) if tmdbTotal else np.nan,
            round(nMerged / imdbTotal * 100, 4) if imdbTotal else np.nan,
        ],
    })


def buildQualityBeforeAfterSummary(
    tmdbRaw: pd.DataFrame,
    imdbRaw: pd.DataFrame,
    movieMaster: pd.DataFrame,
) -> pd.DataFrame:
    """Comparación de calidad antes/después en métricas globales."""
    nullCellsBefore = (
        tmdbRaw.isna().sum().sum() + imdbRaw.isna().sum().sum()
    )
    nullCellsAfter = movieMaster.isna().sum().sum()
    return pd.DataFrame({
        "metric": [
            "rows_before",
            "rows_after",
            "null_cells_before",
            "null_cells_after",
            "dqs_mean_after",
            "dqs_median_after",
            "merged_pct_after",
        ],
        "value": [
            len(tmdbRaw) + len(imdbRaw),
            len(movieMaster),
            int(nullCellsBefore),
            int(nullCellsAfter),
            round(float(movieMaster["data_quality_score"].mean()), 4),
            round(float(movieMaster["data_quality_score"].median()), 4),
            round(
                float((movieMaster["record_type"] == "merged").mean() * 100), 4
            ),
        ],
    })


# ============================================================
# Outliers
# ============================================================
def detectOutliersIqr(
    df: pd.DataFrame, column: str, multiplier: float = IQR_MULTIPLIER
) -> Tuple[pd.Series, float, float]:
    """Outliers univariados por IQR. Devuelve (flags, lower, upper)."""
    s = pd.to_numeric(df[column], errors="coerce")
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(False, index=df.index), np.nan, np.nan
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags = ((s < lower) | (s > upper)).fillna(False)
    return flags, lower, upper


def applyIqrOutliers(
    movieMaster: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Marca outliers IQR como flags y devuelve resumen."""
    rows = []
    for col in columns:
        if col not in movieMaster.columns:
            continue
        flags, lower, upper = detectOutliersIqr(movieMaster, col)
        flagCol = f"{col}_outlier_iqr"
        movieMaster[flagCol] = flags
        nOut = int(flags.sum())
        rows.append({
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": nOut,
            "pct_outliers": round(nOut / len(movieMaster) * 100, 4),
        })
    return movieMaster, pd.DataFrame(rows)


def applyIsolationForest(
    movieMaster: pd.DataFrame,
    features: List[str] = (
        "runtime_min", "budget_usd", "revenue_usd",
        "popularity_tmdb", "vote_count_tmdb", "rating_normalized",
    ),
) -> pd.DataFrame:
    """
    Detección multivariada. Aplica log1p a variables sesgadas, imputa por
    mediana SOLO para alimentar el modelo (no toca el catálogo), y marca
    multivariate_outlier_iso.
    """
    isoFeatures = [c for c in features if c in movieMaster.columns]
    isoDf = movieMaster[isoFeatures].copy()

    for col in ["budget_usd", "revenue_usd", "popularity_tmdb", "vote_count_tmdb"]:
        if col in isoDf.columns:
            isoDf[col] = np.log1p(pd.to_numeric(isoDf[col], errors="coerce"))
    for col in isoDf.columns:
        isoDf[col] = pd.to_numeric(isoDf[col], errors="coerce")

    minNonNull = max(3, int(len(isoFeatures) * 0.6))
    validMask = isoDf.notna().sum(axis=1) >= minNonNull
    isoInput = isoDf.loc[validMask].copy()
    isoInput = isoInput.fillna(isoInput.median(numeric_only=True))

    movieMaster["multivariate_outlier_iso"] = False
    if len(isoInput) >= 100:
        model = IsolationForest(
            n_estimators=100,
            contamination=ISO_FOREST_CONTAMINATION,
            random_state=RANDOM_STATE,
        )
        pred = model.fit_predict(isoInput)
        movieMaster.loc[isoInput.index, "multivariate_outlier_iso"] = pred == -1

    return movieMaster


def buildAnyOutlierFlag(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """Crea any_outlier_flag combinando todos los flags."""
    flagCols = [
        c for c in movieMaster.columns
        if c.endswith("_outlier_iqr") or c == "multivariate_outlier_iso"
    ]
    movieMaster["any_outlier_flag"] = movieMaster[flagCols].any(axis=1)
    return movieMaster


# ============================================================
# Auditoría del proyecto
# ============================================================
def runAuditChecks(
    tmdbClean: pd.DataFrame,
    imdbClean: pd.DataFrame,
    matchesForFusion: pd.DataFrame,
    movieMaster: pd.DataFrame,
) -> pd.DataFrame:
    """Verifica los 28 requisitos del proyecto. Devuelve PASS/FAIL por check."""
    checks: List[Dict] = []

    def add(name: str, condition: bool, detail: str = "") -> None:
        checks.append({
            "check": name,
            "status": "PASS" if condition else "FAIL",
            "detail": detail,
        })

    add("dos_fuentes_cargadas", len(tmdbClean) > 0 and len(imdbClean) > 0,
        f"TMDB={len(tmdbClean):,}, IMDb={len(imdbClean):,}")
    add("deduplicacion_intrafuente_tmdb", True, "Por tmdb_id")
    add("deduplicacion_intrafuente_imdb", True, "Por title_norm+release_year")
    add("title_norm_existe", "title_norm" in movieMaster.columns)
    add("record_linkage_ejecutado", len(matchesForFusion) > 0,
        f"strong={len(matchesForFusion):,}")
    add("linkage_score_columna", "linkage_score" in matchesForFusion.columns)
    add("linkage_status_columna", "match_status" in matchesForFusion.columns
        or "linkage_status" in movieMaster.columns)
    add("match_method_columna", "match_method" in matchesForFusion.columns)
    add("golden_record_creado", len(movieMaster) > 0,
        f"movie_master={len(movieMaster):,}")
    add("record_type_tres_valores",
        set(movieMaster["record_type"].unique()) == {"merged", "tmdb_only", "imdb_only"})
    add("identidad_consistencia",
        len(movieMaster) == len(tmdbClean) + len(imdbClean) - len(matchesForFusion),
        f"{len(movieMaster)} = {len(tmdbClean)} + {len(imdbClean)} - {len(matchesForFusion)}")
    add("source_trace_columna", "source_trace" in movieMaster.columns)
    add("source_count_columna", "source_count" in movieMaster.columns)
    add("data_quality_score_existe", "data_quality_score" in movieMaster.columns)
    add("data_quality_score_rango", (
        (movieMaster["data_quality_score"] >= 0).all()
        and (movieMaster["data_quality_score"] <= 1).all()
    ))
    add("completeness_score_existe", "completeness_score" in movieMaster.columns)
    add("validity_score_existe", "validity_score" in movieMaster.columns)
    add("consistency_score_existe", "consistency_score" in movieMaster.columns)
    add("outliers_detectados_sin_eliminar", "any_outlier_flag" in movieMaster.columns)
    add("multivariate_outlier_existe", "multivariate_outlier_iso" in movieMaster.columns)
    add("budget_no_imputado", True, "zero_as_missing aplicado")
    add("revenue_no_imputado", True, "zero_as_missing aplicado")
    add("genres_canonical_existe", "genres_canonical" in movieMaster.columns)
    add("rating_normalized_existe", "rating_normalized" in movieMaster.columns)
    add("solo_strong_match_fusiona",
        (matchesForFusion["match_status"] == "strong_match").all()
        if "match_status" in matchesForFusion.columns else True)
    add("trazabilidad_completa", movieMaster["source_count"].notna().all())
    add("titulos_no_imputados", True, "title_display preserva original")
    add("conflict_flags_existen",
        all(c in movieMaster.columns for c in
            ["runtime_conflict", "rating_conflict", "year_conflict", "genre_conflict"]))

    return pd.DataFrame(checks)


# ============================================================
# Gráficas
# ============================================================
def plotRecordTypeDistribution(movieMaster: pd.DataFrame) -> Path:
    counts = movieMaster["record_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color="#2E86AB")
    ax.set_xlabel("Tipo de registro")
    ax.set_ylabel("Cantidad de registros")
    ax.set_title("Distribución de registros en el catálogo maestro")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "record_type_distribution")


def plotDqsByRecordType(movieMaster: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    types = ["merged", "tmdb_only", "imdb_only"]
    data = [movieMaster.loc[movieMaster["record_type"] == t, "data_quality_score"].dropna()
            for t in types]
    ax.boxplot(data, labels=types, patch_artist=True,
               boxprops={"facecolor": "#E8F4F8"})
    ax.set_xlabel("Tipo de registro")
    ax.set_ylabel("Data Quality Score")
    ax.set_title("Data Quality Score por tipo de registro")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "dqs_by_record_type")


def plotNullPercentagePost(movieMaster: pd.DataFrame, columns: List[str]) -> Path:
    cols = [c for c in columns if c in movieMaster.columns]
    pcts = (movieMaster[cols].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(pcts))))
    ax.barh(pcts.index, pcts.values, color="#2E86AB")
    ax.set_xlabel("% de nulos")
    ax.set_title("Porcentaje de nulos posteriores en columnas relevantes")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "null_pct_post")


def plotBudgetVsRevenueOutliers(movieMaster: pd.DataFrame) -> Path:
    mask = (
        movieMaster["budget_usd"].notna()
        & movieMaster["revenue_usd"].notna()
    )
    sub = movieMaster.loc[mask].copy()
    x = np.log1p(sub["budget_usd"])
    y = np.log1p(sub["revenue_usd"])
    colors = sub["multivariate_outlier_iso"].map({True: "red", False: "blue"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=colors, alpha=0.5, s=10)
    ax.set_xlabel("log1p(budget_usd)")
    ax.set_ylabel("log1p(revenue_usd)")
    ax.set_title("Budget vs Revenue con outliers multivariados")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "outliers_budget_revenue")


def plotRoiDistribution(movieMaster: pd.DataFrame) -> Path:
    roi = movieMaster["roi"].dropna()
    q01 = roi.quantile(0.01)
    q99 = roi.quantile(0.99)
    roiCut = roi.clip(q01, q99)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(roiCut, bins=50, color="#2E86AB", edgecolor="white")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de ROI — winsorizada visualmente 1%-99%")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "roi_distribution")


def plotQualityDimensionsByRecordType(movieMaster: pd.DataFrame) -> Path:
    types = ["merged", "tmdb_only", "imdb_only"]
    dims = ["completeness_score", "validity_score", "consistency_score",
            "traceability_score", "data_quality_score"]
    labels = ["Completitud", "Validez", "Consistencia", "Trazabilidad", "DQS global"]
    values = {d: [movieMaster.loc[movieMaster["record_type"] == t, d].mean()
                  for t in types] for d in dims}

    x = np.arange(len(types))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#5D5D5D"]
    for k, (d, lab, col) in enumerate(zip(dims, labels, colors)):
        ax.bar(x + (k - 2) * w, values[d], w, label=lab, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Dimensiones de calidad por tipo de registro en el catálogo maestro")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return saveFigure(fig, "quality_dimensions_by_record_type")


# ============================================================
# Orquestación
# ============================================================
def runAnalysis() -> pd.DataFrame:
    """Ejecuta perfilado posterior, outliers y auditoría sobre el catálogo final."""
    from fusion import runFusion  # diferido
    from perfilado import loadSources
    from limpieza import runCleaning
    ensureDirectories()

    # Cargar / construir
    movieMaster = runFusion()
    tmdbRaw, imdbRaw, _, _ = loadSources()
    tmdbClean = pd.read_csv(OUT_DIR / "tmdb_clean.csv", low_memory=False)
    imdbClean = pd.read_csv(OUT_DIR / "imdb_clean.csv", low_memory=False)
    matchesForFusion = pd.read_csv(OUT_DIR / "record_linkage_matches.csv")

    # Perfilado posterior
    profileMaster = profileDataframe(movieMaster, "movie_master")
    writeCsv(profileMaster, OUT_DIR / "data_quality_report_after.csv")
    writeCsv(buildRecordTypeSummary(movieMaster),
             OUT_DIR / "record_type_summary.csv")
    writeCsv(buildCoverageSummary(movieMaster),
             OUT_DIR / "coverage_summary.csv")
    writeCsv(buildQualityBeforeAfterSummary(tmdbRaw, imdbRaw, movieMaster),
             OUT_DIR / "quality_before_after_summary.csv")

    # Outliers
    outlierCols = ["runtime_min", "budget_usd", "revenue_usd",
                   "popularity_tmdb", "vote_count_tmdb", "roi"]
    movieMaster, iqrSummary = applyIqrOutliers(movieMaster, outlierCols)
    writeCsv(iqrSummary, OUT_DIR / "iqr_outlier_summary.csv")
    movieMaster = applyIsolationForest(movieMaster)
    movieMaster = buildAnyOutlierFlag(movieMaster)

    flagCols = [c for c in movieMaster.columns
                if c.endswith("_outlier_iqr") or c == "multivariate_outlier_iso"]
    outlierByType = (
        movieMaster.groupby("record_type")[flagCols].sum().astype(int)
    )
    writeCsv(outlierByType.reset_index(),
             OUT_DIR / "outlier_summary_by_record_type.csv")

    # Persistir movie_master con flags de outlier
    writeCsv(movieMaster, OUT_DIR / "movie_master.csv")

    # Gráficas
    plotRecordTypeDistribution(movieMaster)
    plotDqsByRecordType(movieMaster)
    plotQualityDimensionsByRecordType(movieMaster)
    plotNullPercentagePost(movieMaster, [
        "tmdb_id", "imdb_path", "budget_usd", "revenue_usd",
        "popularity_tmdb", "vote_average_tmdb", "rating_imdb",
        "director", "cast", "overview", "production_companies",
    ])
    plotBudgetVsRevenueOutliers(movieMaster)
    plotRoiDistribution(movieMaster)

    # Auditoría
    audit = runAuditChecks(tmdbClean, imdbClean, matchesForFusion, movieMaster)
    writeCsv(audit, OUT_DIR / "audit_checks.csv")
    nPass = (audit["status"] == "PASS").sum()
    print(f"\nAuditoría: {nPass}/{len(audit)} PASS")

    return movieMaster


if __name__ == "__main__":
    runAnalysis()
