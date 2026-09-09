# Media Catalog Intelligence

**Pipeline reproducible de calidad de datos e integración de dos fuentes cinematográficas
(TMDB + IMDb) sin identificador común, bajo el marco TDQM (Total Data Quality Management).**

El objetivo no es "juntar dos CSV": es **integrarlos de forma defendible y auditable**, produciendo
un catálogo maestro (`movie_master`) donde cada dato conserva la trazabilidad de qué fuente lo aportó,
tiene un score de calidad y queda marcado cuando las fuentes se contradicen.

> 📄 **Reporte técnico completo (47 pp.):** [`docs/Reporte_Final.pdf`](docs/Reporte_Final.pdf)

---

## El problema

Las plataformas de *streaming*, distribuidoras y estudios toman decisiones financieras y de
mercadotecnia sobre metadatos que viven repartidos en fuentes distintas. Esas fuentes:

- usan **identificadores propios** que no son comparables (`tmdb_id` numérico vs. rutas `/title/tt…/`),
- nombran los campos distinto (`Generes`, `Plot Kyeword`) y con **errores de captura**,
- guardan el mismo atributo con **formatos heterogéneos** (`"2 hours 27 minutes"`, `"142"`, `"not-released"`),
- arrastran problemas sistemáticos: el **90 % de los años de IMDb** están almacenados con signo negativo;
  el campo `Run Time` de IMDb viene contaminado con presupuestos y *ratings*.

Un `merge` de pandas no resuelve nada de esto.

---

## Qué hace el pipeline

| Paso | Módulo | Salida |
|------|--------|--------|
| 1. **Perfilado** inicial y diagnóstico por fuente | `perfilado.py` | reporte de calidad por columna, diagnósticos específicos |
| 2. **Limpieza** intra-fuente por tipo de campo (texto, fecha, duración, dinero, conteos, géneros) | `limpieza.py` | `tmdb_clean.csv`, `imdb_clean.csv` |
| 3. **Record linkage** probabilístico + **Golden Record** | `fusion.py` | `movie_master.csv`, archivos de auditoría de *matches* |
| 4. **Perfilado posterior + outliers + auditoría** | `analisis_calidad.py` | `movie_master.csv` con flags, `audit_checks.csv` |
| 5. **Análisis**: consultas, modelos, recomendador, rankings y conclusiones | `analisis.py` + `consultas_*.py` | 40+ CSV y 30+ figuras |

### Record linkage en cascada

1. **Match exacto** por `(título_normalizado, año)`
2. **Título exacto** con tolerancia de año ±1
3. **Fuzzy matching** (`rapidfuzz`, `token_set_ratio`) **bloqueado** por `(primera letra, año)` para
   acotar el espacio de comparación de ~946 k × 24 k a algo tratable

Cada par recibe un `linkage_score` compuesto (título 60 % · año 25 % · duración 10 % · género 5 %,
renormalizado según las señales disponibles) y se resuelve **greedy 1-a-1**. Solo los
`strong_match` (score ≥ 90) se fusionan; los `probable_match` y `manual_review` se persisten como
evidencia, **no** se fusionan.

### Decisiones de diseño

- **Nunca se imputan `budget` ni `revenue`.** Un valor sintético contaminaría los modelos y los
  rankings. Regla: *exactitud operativa > completitud cosmética.*
- **El cero financiero se interpreta como faltante** (`zero_as_missing`): $0 de presupuesto no
  significa que la película fuera gratuita, significa que el dato no se reportó.
- **Los outliers no se eliminan**, se marcan con flags (IQR univariado + Isolation Forest multivariado).
- **Trazabilidad a nivel de campo** (`source_trace`, `source_count`) y **conflict flags** cuando
  TMDB e IMDb no coinciden en duración, *rating*, año o género.
- **Data Quality Score por registro**: media ponderada de completitud, validez, consistencia,
  unicidad y trazabilidad.

---

## Resultados (sobre el catálogo integrado)

- **946 460** registros TMDB + **24 402** IMDb → catálogo maestro unificado.
- **19 862** fusiones fuertes (`strong_match`); `linkage_score` medio **99.6**.
- Tasa de fusión sobre IMDb: **83 %** (reconstruida por similitud, sin ID común).
- `733` *probable* + `238` *manual review* aislados para revisión humana.
- Modelos sobre el catálogo: clasificación de rentabilidad ROC-AUC ≈ **0.96**,
  regresión de *revenue* R²(log) ≈ **0.96** (RandomForest).
- Recomendador de contenido: TF-IDF (géneros + *keywords* + *overview*) + NearestNeighbors coseno.

> **Nota de honestidad analítica:** los valores `budget`/`revenue` de este *dataset* de TMDB son
> sintéticos y varias señales de los modelos son post-estreno. Las cifras absolutas deben leerse como
> **relativas dentro del catálogo**, no como pronósticos en USD. Esto se documenta en la Sección 11
> del reporte.

---

## Stack

Python 3.11 · pandas · NumPy · scikit-learn · rapidfuzz · matplotlib

---

## Cómo ejecutar

```bash
pip install -r requirements.txt
python main.py                # pipeline completo
python main.py --step 1-3     # solo perfilado → fusión
python main.py --from 4       # desde outliers/auditoría (asume pasos previos hechos)
```

Los CSV fuente **no se incluyen** (`movies.csv` supera el límite de 100 MB de GitHub). Descárgalos
y colócalos en `data/`; el pipeline los detecta por nombre:

| Fuente | Nombres aceptados |
|--------|-------------------|
| TMDB | `movies.csv` · `tmdb_movies.csv` · `movies_metadata_cleaned_1900_2025.csv` |
| IMDb | `25k IMDb movie Dataset.csv` · `25k_imdb_movie_dataset.csv` · `imdb_movies.csv` |

Los resultados se escriben en `outputs/` (CSV) y `figures/` (PNG), que se crean automáticamente.

---

## Estructura

```
├── main.py                   # punto de entrada único
├── config.py                 # rutas, umbrales y constantes
├── utils.py                  # helpers de I/O, normalización y gráficas
├── perfilado.py              # paso 1
├── limpieza.py               # paso 2
├── fusion.py                 # paso 3 — record linkage + Golden Record
├── analisis_calidad.py       # paso 4 — outliers + auditoría
├── analisis.py               # paso 5 — modelos, recomendador, conclusiones
├── consultas_tmdb.py         # 5 consultas exclusivas TMDB
├── consultas_imdb.py         # 5 consultas exclusivas IMDb
├── consultas_multifuente.py  # 10 consultas integradas
└── docs/Reporte_Final.pdf    # reporte técnico completo
```

---

## Autoría

Proyecto final — *Calidad y Preprocesamiento de Datos*, Ciencia de Datos, UNAM (IIMAS).
Equipo: Ashley Yael López Espinoza · Denzel Gael Cruz Prieto · Gustavo Mier Basilio ·
Pedro Manuel Cardón Carrillo · Salma Annette Rodríguez Muñoz.
