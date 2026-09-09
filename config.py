"""
config.py
---------
Rutas, constantes y umbrales del pipeline de calidad de datos.
No importa otros módulos del proyecto: solo expone configuración.
"""

from pathlib import Path

# ============================================================
# Reproducibilidad
# ============================================================
RANDOM_STATE = 42

# ============================================================
# Estructura de directorios
# ============================================================
BASE_DIR = Path(".").resolve()

DATA_DIRS = [Path("."), Path("./data"), Path("/content"), Path("/mnt/data")]

OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
REP_DIR = BASE_DIR / "reports"


def ensureDirectories() -> None:
    """Crea las carpetas estándar del proyecto si no existen."""
    for d in [OUT_DIR, FIG_DIR, REP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Nombres candidatos para las fuentes
# ============================================================
TMDB_CANDIDATE_FILES = [
    "movies.csv",
    "tmdb_movies.csv",
    "movies_metadata_cleaned_1900_2025.csv",
]

IMDB_CANDIDATE_FILES = [
    "25k IMDb movie Dataset.csv",
    "25k_imdb_movie_dataset.csv",
    "imdb_movies.csv",
]


# ============================================================
# Umbrales del record linkage
# ============================================================
SCORE_STRONG_MIN = 90.0     # >= 90 fusiona
SCORE_PROBABLE_MIN = 82.0   # [82, 90) revisión
SCORE_MANUAL_MIN = 75.0     # [75, 82) manual review; < 75 descartado

# Pesos del linkage_score compuesto
WEIGHT_TITLE = 0.60
WEIGHT_YEAR = 0.25
WEIGHT_RUNTIME = 0.10
WEIGHT_GENRE = 0.05


# ============================================================
# Umbrales de conflict flags
# ============================================================
CONFLICT_RUNTIME_DELTA_MIN = 10  # minutos
CONFLICT_RATING_DELTA = 2.0      # puntos
CONFLICT_YEAR_DELTA = 1          # años


# ============================================================
# Pesos del data_quality_score
# ============================================================
DQS_WEIGHT_COMPLETENESS = 0.35
DQS_WEIGHT_VALIDITY = 0.25
DQS_WEIGHT_CONSISTENCY = 0.20
DQS_WEIGHT_UNIQUENESS = 0.10
DQS_WEIGHT_TRACEABILITY = 0.10


# ============================================================
# Detección de outliers
# ============================================================
ISO_FOREST_CONTAMINATION = 0.03
IQR_MULTIPLIER = 1.5


# ============================================================
# Modelos predictivos
# ============================================================
TEST_SIZE = 0.25
N_ESTIMATORS_RF = 100
MAX_DEPTH_RF = None


# ============================================================
# Mapeo canónico de géneros
# ============================================================
GENRE_MAP = {
    "sci-fi": "Science Fiction",
    "sci fi": "Science Fiction",
    "science-fiction": "Science Fiction",
    "science fiction": "Science Fiction",
    "rom-com": "Romance",
    "romcom": "Romance",
    "biography": "Biographical",
    "action adventure": "Action",
    "tv movie": "TV Movie",
}
