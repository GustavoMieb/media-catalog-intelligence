# Proyecto de Calidad de Datos — Películas TMDB + IMDb

Pipeline modular de perfilado, limpieza, fusión y análisis de dos fuentes de datos cinematográficos.

---

## Estructura del proyecto

```
proyecto/
│
├── main.py                   # Punto de entrada único (ejecutar esto)
│
├── config.py                 # Rutas, constantes y umbrales globales
├── utils.py                  # Helpers compartidos (I/O, normalización, gráficas)
│
├── perfilado.py              # Paso 1 — Carga y perfilado inicial de las fuentes
├── limpieza.py               # Paso 2 — Limpieza intra-fuente por campo
├── fusion.py                 # Paso 3 — Record linkage + Golden Record (movie_master)
├── analisis_calidad.py       # Paso 4 — Perfilado posterior, outliers y auditoría
├── analisis.py               # Paso 5 — Consultas, modelos, recomendador y conclusiones
│
├── consultas_tmdb.py         # 5 consultas exclusivas sobre registros TMDB
├── consultas_imdb.py         # 5 consultas exclusivas sobre registros IMDb
├── consultas_multifuente.py  # 10 consultas integradas TMDB + IMDb
│
├── data/                     # Carpeta sugerida para los CSV fuente
├── outputs/                  # CSVs generados (se crea automáticamente)
├── figures/                  # Gráficas PNG (se crea automáticamente)
└── reports/                  # Reportes adicionales (se crea automáticamente)
```

---

## Requisitos

### Python
Versión **3.9 o superior**.

### Dependencias
```bash
pip install -r requirements.txt
```

---

## Datos de entrada

> Los CSV fuente **no se incluyen en el repositorio** por tamaño (`movies.csv` supera el
> límite de 100 MB de GitHub). Descárgalos y colócalos en la carpeta `data/` (o en la raíz
> del proyecto). El sistema los detecta automáticamente por nombre.

| Fuente | Nombres de archivo aceptados |
|--------|------------------------------|
| TMDB   | `movies.csv` · `tmdb_movies.csv` · `movies_metadata_cleaned_1900_2025.csv` |
| IMDb   | `25k IMDb movie Dataset.csv` · `25k_imdb_movie_dataset.csv` · `imdb_movies.csv` |

---

## Cómo ejecutar

### Pipeline completo (recomendado)
```bash
python main.py
```


---

## Descripción de los módulos

### `perfilado.py`
Carga las dos fuentes desde disco, detecta el archivo correcto automáticamente y genera un reporte de calidad por columna (tipo, nulos, unicidad) para cada fuente. Incluye diagnósticos específicos: años negativos en IMDb, revenue/budget en cero, formatos inválidos de runtime.

### `limpieza.py`
Aplica limpieza especializada por tipo de campo: normalización de títulos (clave de comparación), parseo robusto de runtime (formatos HH:MM, lenguaje natural, `not-released`), parseo de dinero con detección de ceros como faltantes, normalización canónica de géneros y deduplicación intra-fuente.

### `fusion.py`
Record linkage en cascada de tres capas:
1. Match exacto por `(title_norm, release_year)`
2. Título exacto con año ±1
3. Fuzzy bloqueado por `(primera_letra, año)`

Solo los matches con `linkage_score ≥ 90` se fusionan en el Golden Record. Los probable y manual_review se persisten para auditoría.

### `analisis_calidad.py`
Perfilado posterior del catálogo maestro. Detecta outliers univariados (IQR) y multivariados (Isolation Forest) sin eliminar registros — solo los marca con flags. Ejecuta 28 checks de auditoría técnica del proyecto.

### `analisis.py`
Análisis completo de negocio:
- **Consultas**: 5 por TMDB, 5 por IMDb, 10 integradas multi-fuente
- **Modelos predictivos**: clasificación de rentabilidad (RandomForest vs LogisticRegression) y predicción de revenue (RandomForest vs GradientBoosting vs Ridge)
- **Recomendador**: TF-IDF + NearestNeighbors con distancia coseno sobre géneros + keywords + overview
- **Rankings**: géneros para inversión (strategic_investment_score) y películas para marketing (marketing_priority_score)
- **Conclusiones**: técnicas y de negocio exportadas a CSV y TXT

