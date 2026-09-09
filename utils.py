"""
utils.py
--------
Funciones auxiliares compartidas entre los módulos del proyecto.
Sin lógica de negocio: solo helpers reutilizables.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIRS, FIG_DIR


# ============================================================
# Entrada / salida
# ============================================================
def findFile(candidates: Iterable[str], dataDirs: Iterable[Path] = DATA_DIRS) -> Path:
    """Busca el primer archivo existente entre los candidatos en los directorios dados."""
    for folder in dataDirs:
        for name in candidates:
            candidate = folder / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No encontré ninguno de estos archivos: {list(candidates)}")


def saveFigure(fig: plt.Figure, name: str, dpi: int = 130, tight: bool = True) -> Path:
    """Guarda una figura en FIG_DIR con nombre estandarizado y la cierra."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outPath = FIG_DIR / (name if name.endswith(".png") else f"{name}.png")
    if tight:
        fig.savefig(outPath, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(outPath, dpi=dpi)
    plt.close(fig)
    return outPath


def writeCsv(df: pd.DataFrame, outPath: Path, index: bool = False) -> Path:
    """Guarda un DataFrame como CSV asegurando que el directorio existe."""
    outPath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outPath, index=index)
    return outPath


# ============================================================
# Normalización de cadenas
# ============================================================
def snakeCase(text: str) -> str:
    """Convierte un nombre de columna a snake_case ASCII."""
    text = str(text).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def removeAccents(text: Optional[str]) -> Optional[str]:
    """Elimina acentos vía descomposición Unicode (NFD)."""
    if pd.isna(text):
        return np.nan
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def normalizeBasicText(value: Optional[str]) -> Optional[str]:
    """Normalización básica: minúsculas, sin acentos, alfanumérico + espacios."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else np.nan


def snakeCaseColumns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra todas las columnas de un DataFrame a snake_case (in-place safe)."""
    df.columns = [snakeCase(c) for c in df.columns]
    return df


# ============================================================
# Pretty-print de tablas
# ============================================================
def summarizeRows(df: pd.DataFrame, label: str = "df") -> str:
    """Devuelve un string de resumen de un DataFrame: filas, cols, nulos."""
    n = len(df)
    cols = df.shape[1]
    nulls = int(df.isna().sum().sum())
    return f"{label}: {n:,} filas × {cols} cols ({nulls:,} celdas nulas)"


def topValueCounts(series: pd.Series, n: int = 10) -> pd.DataFrame:
    """Top-N de value_counts con frecuencia y porcentaje."""
    vc = series.value_counts(dropna=False).head(n)
    pct = (vc / len(series) * 100).round(2)
    return pd.DataFrame({"value": vc.index, "count": vc.values, "pct": pct.values})


# ============================================================
# Helpers de listas y gráficas compartidas entre consultas
# ============================================================
def ensureList(x) -> list:
    """Garantiza que un valor sea lista, parseando string serializado si es necesario."""
    import ast
    import re
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s in ["", "nan", "None", "[]"]:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [p.strip() for p in re.split(r"[,;|]", s) if p.strip()]
    return []


def plotBarh(df: pd.DataFrame, xCol: str, yCol: str, title: str,
             xlabel: str, ylabel: str, fname: str) -> Path:
    """Gráfica de barras horizontales estándar del proyecto."""
    from config import FIG_DIR
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df[yCol].astype(str), df[xCol], color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return saveFigure(fig, fname)


def plotBar(df: pd.DataFrame, xCol: str, yCol: str, title: str,
            xlabel: str, ylabel: str, fname: str, rotation: int = 30) -> Path:
    """Gráfica de barras verticales estándar del proyecto."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[xCol].astype(str), df[yCol], color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return saveFigure(fig, fname)
