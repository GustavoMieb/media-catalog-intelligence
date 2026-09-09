"""
analisis.py
-----------
Análisis completo del catálogo maestro movie_master.csv:
    - Consultas por fuente (TMDB e IMDb) y consultas integradas
    - Modelos predictivos (rentabilidad y revenue)
    - Sistema de recomendación basado en contenido (TF-IDF + NearestNeighbors)
    - Ranking estratégico de géneros para inversión
    - Priorización de películas para marketing
    - Conclusiones de negocio y técnicas

Depende de: outputs/movie_master.csv (generado por analisis_calidad.py)

Salidas:
    outputs/model_profitability_results.csv
    outputs/model_revenue_results.csv
    outputs/genre_investment_ranking.csv
    outputs/marketing_priority_ranking.csv
    outputs/final_conclusions.csv
    outputs/final_report_summary.txt
    figures/model_profitability_*.png
    figures/model_revenue_*.png
    figures/genre_investment_ranking.png
    figures/marketing_priority_ranking.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    MAX_DEPTH_RF,
    N_ESTIMATORS_RF,
    OUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    ensureDirectories,
)
from utils import ensureList, plotBar, plotBarh, saveFigure, writeCsv


# ============================================================
# Helpers
# ============================================================
def loadMaster() -> pd.DataFrame:
    """Carga el catálogo maestro desde disco."""
    path = OUT_DIR / "movie_master.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta analisis_calidad.py primero."
        )
    return pd.read_csv(path, low_memory=False)


# ============================================================
# Bloque 11. Modelos predictivos
# ============================================================
def _prepareFeatures(
    df: pd.DataFrame,
    targetCol: str,
    featureCols: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Filtra nulos, selecciona features numéricas disponibles y devuelve X, y."""
    available = [c for c in featureCols if c in df.columns]
    sub = df[[targetCol] + available].dropna(subset=[targetCol])
    sub = sub.copy()
    for c in available:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=available, how="all")
    sub[available] = sub[available].fillna(sub[available].median(numeric_only=True))
    if len(sub) < 50:
        return None, None
    return sub[available], sub[targetCol]


def runProfitabilityModel(movieMaster: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Clasifica si una película será rentable (is_profitable).
    Compara RandomForest vs LogisticRegression.
    """
    featureCols = [
        "budget_usd", "runtime_min", "popularity_tmdb",
        "vote_count_tmdb", "release_year",
    ]
    X, y = _prepareFeatures(movieMaster, "is_profitable", featureCols)
    if X is None:
        print("Modelo rentabilidad: datos insuficientes.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, prob), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

    resultsDf = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    writeCsv(resultsDf, OUT_DIR / "model_profitability_results.csv")

    # Gráfica comparativa
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(resultsDf))
    w = 0.25
    for k, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        ax.bar(x + k * w, resultsDf[metric], w, label=metric)
    ax.set_xticks(x + w)
    ax.set_xticklabels(resultsDf["model"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Modelos de rentabilidad — comparativa de métricas")
    ax.set_ylabel("Score")
    ax.legend()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    saveFigure(fig, "model_profitability_comparison")

    print("\nModelos de rentabilidad:")
    print(resultsDf.to_string(index=False))
    return resultsDf


def runRevenueModel(movieMaster: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Predice revenue_usd con log1p.
    Compara RandomForestRegressor vs GradientBoosting vs Ridge.
    """
    sub = movieMaster[movieMaster["revenue_usd"].notna()
                      & movieMaster["budget_usd"].notna()].copy()
    sub["log_revenue"] = np.log1p(sub["revenue_usd"])
    sub["log_budget"] = np.log1p(sub["budget_usd"])

    featureCols = [
        "log_budget", "runtime_min", "popularity_tmdb",
        "vote_count_tmdb", "release_year",
    ]
    X, y = _prepareFeatures(sub, "log_revenue", featureCols)
    if X is None:
        print("Modelo revenue: datos insuficientes.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH_RF,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Ridge": Ridge(alpha=1.0),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "model": name,
            "r2": round(r2_score(y_test, pred), 4),
            "mae_log": round(mean_absolute_error(y_test, pred), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

    resultsDf = pd.DataFrame(results).sort_values("r2", ascending=False)
    writeCsv(resultsDf, OUT_DIR / "model_revenue_results.csv")

    # Gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(resultsDf["model"], resultsDf["r2"], color="#2E86AB")
    ax.set_title("Modelos de revenue — R² en test")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    saveFigure(fig, "model_revenue_r2")

    print("\nModelos de revenue:")
    print(resultsDf.to_string(index=False))
    return resultsDf


# ============================================================
# Bloque 12. Sistema de recomendación (TF-IDF + NearestNeighbors)
# ============================================================
def buildRecommender(
    movieMaster: pd.DataFrame, n_neighbors: int = 11
) -> Tuple[NearestNeighbors, pd.DataFrame, TfidfVectorizer]:
    """
    Construye un recomendador basado en contenido.
    Texto = géneros + keywords + overview (truncado).
    """
    sub = movieMaster[movieMaster["title_display"].notna()].copy().reset_index(drop=True)

    def buildText(row) -> str:
        parts = []
        genres = ensureList(row.get("genres_list", []))
        parts.extend(genres)
        kw = str(row.get("keywords", "")).strip()
        if kw and kw not in ["nan", ""]:
            parts.append(kw[:200])
        ov = str(row.get("overview", "")).strip()
        if ov and ov not in ["nan", ""]:
            parts.append(ov[:300])
        return " ".join(parts)

    sub["content_text"] = sub.apply(buildText, axis=1)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(sub["content_text"])

    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(sub)),
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(matrix)

    print(f"Recomendador construido: {len(sub):,} películas, {matrix.shape[1]:,} features TF-IDF")
    return nn, sub, vec


def recommend(
    title: str,
    nn: NearestNeighbors,
    catalog: pd.DataFrame,
    vec: TfidfVectorizer,
    n: int = 5,
) -> pd.DataFrame:
    """Devuelve las n películas más similares a `title`."""
    matches = catalog[catalog["title_display"].str.lower() == title.lower()]
    if len(matches) == 0:
        print(f"Película '{title}' no encontrada en el catálogo.")
        return pd.DataFrame()
    idx = matches.index[0]
    content = catalog.loc[idx, "content_text"]
    vector = vec.transform([content])
    distances, indices = nn.kneighbors(vector, n_neighbors=n + 1)
    recs = catalog.iloc[indices[0][1:]].copy()
    recs["similarity_score"] = (1 - distances[0][1:]).round(4)
    return recs[["title_display", "release_year", "genres_canonical",
                 "rating_normalized", "similarity_score"]].reset_index(drop=True)


# ============================================================
# Bloque 13. Ranking estratégico de géneros para inversión
# ============================================================
def buildGenreInvestmentRanking(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking de géneros con un strategic_investment_score compuesto por:
    rentabilidad, revenue, popularidad, rating, actividad reciente,
    cobertura financiera, outliers y calidad de datos.
    """
    sub = movieMaster.copy()
    sub["genres_list"] = sub["genres_list"].apply(ensureList)
    exploded = sub.explode("genres_list")
    exploded = exploded[exploded["genres_list"] != "Unknown"]

    agg = (
        exploded.groupby("genres_list")
        .agg(
            n_movies=("title_display", "count"),
            profitable_pct=("is_profitable", "mean"),
            revenue_mean=("revenue_usd", "mean"),
            revenue_median=("revenue_usd", "median"),
            popularity_mean=("popularity_tmdb", "mean"),
            rating_mean=("rating_normalized", "mean"),
            dqs_mean=("data_quality_score", "mean"),
            recent_pct=("release_year", lambda x: (x >= 2015).mean()),
            budget_coverage=("budget_usd", lambda x: x.notna().mean()),
            outlier_pct=("any_outlier_flag", "mean"),
        )
        .reset_index()
        .rename(columns={"genres_list": "genre"})
    )
    agg = agg[agg["n_movies"] >= 30].copy()

    # Normalizar 0-1
    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    agg["strategic_investment_score"] = (
        0.30 * minmax(agg["profitable_pct"].fillna(0))
        + 0.20 * minmax(agg["revenue_median"].fillna(0))
        + 0.15 * minmax(agg["popularity_mean"].fillna(0))
        + 0.10 * minmax(agg["rating_mean"].fillna(0))
        + 0.10 * minmax(agg["recent_pct"].fillna(0))
        + 0.05 * minmax(agg["budget_coverage"].fillna(0))
        + 0.05 * minmax(agg["dqs_mean"].fillna(0))
        - 0.05 * minmax(agg["outlier_pct"].fillna(0))
    ).round(4)

    ranking = agg.sort_values("strategic_investment_score", ascending=False).reset_index(drop=True)
    writeCsv(ranking, OUT_DIR / "genre_investment_ranking.csv")

    top15 = ranking.head(15)
    plotBarh(
        top15.sort_values("strategic_investment_score"),
        xCol="strategic_investment_score",
        yCol="genre",
        title="Ranking estratégico de géneros para inversión (top 15)",
        xlabel="Strategic Investment Score",
        ylabel="Género",
        fname="genre_investment_ranking",
    )

    print("\nTop 10 géneros para inversión:")
    print(ranking[["genre", "n_movies", "strategic_investment_score"]].head(10).to_string(index=False))
    return ranking


# ============================================================
# Bloque 14. Priorización de películas para marketing
# ============================================================
def buildMarketingPriorityRanking(movieMaster: pd.DataFrame) -> pd.DataFrame:
    """
    Rankea películas del catálogo por marketing_priority_score:
    potencial comercial × calidad × visibilidad, sin conflictos inter-fuente.
    """
    confCols = ["runtime_conflict", "rating_conflict", "year_conflict", "genre_conflict"]
    sub = movieMaster.copy()

    # Excluir registros con conflictos si las columnas existen
    if all(c in sub.columns for c in confCols):
        conflict_mask = sub[confCols].fillna(False).any(axis=1)
        sub = sub[~conflict_mask]

    sub = sub[
        sub["revenue_usd"].notna()
        & sub["rating_normalized"].notna()
        & sub["popularity_tmdb"].notna()
    ].copy()

    if len(sub) == 0:
        print("Marketing ranking: sin registros suficientes.")
        return pd.DataFrame()

    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    sub["marketing_priority_score"] = (
        0.40 * minmax(np.log1p(sub["revenue_usd"]))
        + 0.30 * minmax(sub["rating_normalized"])
        + 0.20 * minmax(sub["popularity_tmdb"])
        + 0.10 * minmax(sub["data_quality_score"].fillna(0))
    ).round(4)

    ranking = (
        sub.sort_values("marketing_priority_score", ascending=False)
        .head(50)[["title_display", "release_year", "revenue_usd",
                   "rating_normalized", "popularity_tmdb",
                   "genres_canonical", "marketing_priority_score"]]
        .reset_index(drop=True)
    )
    writeCsv(ranking, OUT_DIR / "marketing_priority_ranking.csv")

    top20 = ranking.head(20)
    plotBarh(
        top20.sort_values("marketing_priority_score"),
        xCol="marketing_priority_score",
        yCol="title_display",
        title="Priorización de películas para marketing (top 20)",
        xlabel="Marketing Priority Score",
        ylabel="Película",
        fname="marketing_priority_ranking",
    )

    print("\nTop 10 películas para marketing:")
    print(ranking[["title_display", "marketing_priority_score"]].head(10).to_string(index=False))
    return ranking


# ============================================================
# Bloque 15. Conclusiones
# ============================================================
def buildConclusions(
    movieMaster: pd.DataFrame,
    profitModel: Optional[pd.DataFrame],
    revenueModel: Optional[pd.DataFrame],
    genreRanking: pd.DataFrame,
    marketingRanking: pd.DataFrame,
) -> pd.DataFrame:
    """Genera conclusiones técnicas y de negocio estructuradas."""
    nMaster = len(movieMaster)
    nMerged = int((movieMaster["record_type"] == "merged").sum())
    nTmdbOnly = int((movieMaster["record_type"] == "tmdb_only").sum())
    nImdbOnly = int((movieMaster["record_type"] == "imdb_only").sum())
    avgDqs = round(float(movieMaster["data_quality_score"].mean()), 4)
    nOutliers = int(movieMaster.get("any_outlier_flag", pd.Series(dtype=bool)).sum())

    best_profit = profitModel.iloc[0] if profitModel is not None and len(profitModel) > 0 else None
    best_revenue = revenueModel.iloc[0] if revenueModel is not None and len(revenueModel) > 0 else None
    best_genre = genreRanking.iloc[0] if len(genreRanking) > 0 else None
    best_marketing = marketingRanking.iloc[0] if len(marketingRanking) > 0 else None

    conclusions = [
        f"El catálogo maestro integró {nMaster:,} registros únicos de películas: "
        f"{nMerged:,} merges fuertes, {nTmdbOnly:,} exclusivos TMDB, {nImdbOnly:,} exclusivos IMDb.",

        f"El Data Quality Score promedio del catálogo es {avgDqs}, lo que indica el nivel general "
        f"de completitud, validez, consistencia, unicidad y trazabilidad tras el pipeline.",

        f"Se detectaron {nOutliers:,} registros con al menos un flag de outlier (IQR o Isolation Forest). "
        f"Ninguno fue eliminado; se preservaron con flags para análisis diferenciado.",

        "El perfilado inicial reveló problemas sistemáticos en ambas fuentes: años negativos en IMDb, "
        "revenue y budget en cero (interpretados como faltantes), y formatos heterogéneos de runtime.",

        "El record linkage utilizó una cascada de tres capas (exact, pm1, fuzzy) con bloqueo por "
        "primera letra y año, garantizando cobertura sin sacrificar precisión.",
    ]

    if best_profit is not None:
        conclusions.append(
            f"El mejor modelo de rentabilidad fue {best_profit['model']} con ROC-AUC={best_profit['roc_auc']} "
            f"y F1={best_profit['f1']}. Permite identificar películas potencialmente rentables antes del estreno."
        )

    if best_revenue is not None:
        conclusions.append(
            f"El mejor modelo de predicción de revenue fue {best_revenue['model']} con R²={best_revenue['r2']}. "
            f"Trabajar en escala log1p mitigó el sesgo de las películas de altísimo recaudo."
        )

    conclusions.append(
        "El sistema de recomendación basado en TF-IDF + NearestNeighbors con distancia coseno permite "
        "recomendar películas similares sin requerir datos de usuarios (filtrado colaborativo)."
    )

    if best_genre is not None:
        conclusions.append(
            f"El género con mayor strategic_investment_score fue '{best_genre['genre']}' "
            f"(score={best_genre['strategic_investment_score']}), considerando rentabilidad, revenue, "
            f"popularidad, actividad reciente y calidad de datos."
        )

    if best_marketing is not None:
        conclusions.append(
            f"La película con mayor prioridad de marketing fue '{best_marketing['title_display']}' "
            f"(score={best_marketing['marketing_priority_score']}), con alto revenue, rating y popularidad "
            f"sin conflictos inter-fuente."
        )

    conclusions.append(
        "En conjunto, el proyecto demuestra que la calidad de datos es una condición necesaria para "
        "análisis confiables: sin limpieza, validación y fusión controlada, los modelos y rankings "
        "estarían sesgados por duplicados, inconsistencias y valores atípicos."
    )

    df = pd.DataFrame({
        "conclusion_id": [f"C{i+1}" for i in range(len(conclusions))],
        "type": (
            ["tecnica"] * 5
            + (["modelo"] if best_profit is not None else [])
            + (["modelo"] if best_revenue is not None else [])
            + ["tecnica", "negocio", "negocio", "negocio"]
        )[:len(conclusions)],
        "conclusion": conclusions,
    })

    writeCsv(df, OUT_DIR / "final_conclusions.csv")
    with open(OUT_DIR / "final_report_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(conclusions))

    print("\n=== CONCLUSIONES ===")
    for c in conclusions:
        print(f"  • {c}")

    return df


# ============================================================
# Orquestación principal
# ============================================================
def runAnalysis() -> None:
    """
    Ejecuta el pipeline completo de análisis.
    Requiere outputs/movie_master.csv (producido por analisis_calidad.py).
    """
    ensureDirectories()

    print("=== Cargando catálogo maestro ===")
    movieMaster = loadMaster()
    print(f"  {len(movieMaster):,} registros cargados.")

    # Consultas por fuente
    print("\n=== Consultas TMDB ===")
    from consultas_tmdb import runTmdbQueries
    runTmdbQueries(movieMaster)

    print("\n=== Consultas IMDb ===")
    from consultas_imdb import runImdbQueries
    runImdbQueries(movieMaster)

    print("\n=== Consultas integradas (multi-fuente) ===")
    from consultas_multifuente import runIntegratedQueries
    runIntegratedQueries(movieMaster)

    # Modelos predictivos
    print("\n=== Modelos predictivos ===")
    profitModel = runProfitabilityModel(movieMaster)
    revenueModel = runRevenueModel(movieMaster)

    # Recomendador
    print("\n=== Sistema de recomendación ===")
    nn, catalog, vec = buildRecommender(movieMaster)
    # Ejemplo de prueba con la primera película disponible
    sampleTitle = catalog["title_display"].dropna().iloc[0]
    recs = recommend(sampleTitle, nn, catalog, vec)
    if len(recs) > 0:
        print(f"  Recomendaciones para '{sampleTitle}':")
        print(recs.to_string(index=False))

    # Rankings
    print("\n=== Ranking de géneros para inversión ===")
    genreRanking = buildGenreInvestmentRanking(movieMaster)

    print("\n=== Priorización de películas para marketing ===")
    marketingRanking = buildMarketingPriorityRanking(movieMaster)

    # Conclusiones
    print("\n=== Conclusiones ===")
    buildConclusions(movieMaster, profitModel, revenueModel, genreRanking, marketingRanking)

    print("\n✓ analisis.py completado. Resultados en outputs/ y figures/")


if __name__ == "__main__":
    runAnalysis()
