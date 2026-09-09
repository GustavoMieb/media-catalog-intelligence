"""
main.py
-------
Punto de entrada único del pipeline de calidad de datos y análisis de películas.

Orden de ejecución:
    1. perfilado.py       → diagnóstico y perfilado inicial de TMDB e IMDb
    2. limpieza.py        → limpieza intra-fuente → tmdb_clean.csv, imdb_clean.csv
    3. fusion.py          → record linkage + Golden Record → movie_master.csv (sin outliers)
    4. analisis_calidad.py→ perfilado posterior + outliers + auditoría → movie_master.csv (completo)
    5. analisis.py        → consultas + modelos + recomendador + rankings + conclusiones

Uso:
    python main.py              # Pipeline completo
    python main.py --step 1     # Solo perfilado
    python main.py --step 1-3   # Pasos 1 a 3
    python main.py --from 4     # Desde el paso 4 (asume pasos anteriores ya ejecutados)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

# Fuerza UTF-8 en la salida estándar para que los símbolos del log (✓, ✗, •, ×)
# no rompan la ejecución cuando stdout está redirigido a un archivo o pipe en
# Windows (consola cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de calidad de datos y análisis de películas TMDB + IMDb"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step", type=str, default=None,
        help="Paso(s) a ejecutar. Ej: '1', '1-3', '2,4'"
    )
    group.add_argument(
        "--from", dest="from_step", type=int, default=None,
        help="Ejecutar desde este paso hasta el final"
    )
    return parser.parse_args()


def resolveSteps(args: argparse.Namespace, totalSteps: int = 5) -> List[int]:
    """Determina qué pasos ejecutar según los argumentos."""
    if args.step is None and args.from_step is None:
        return list(range(1, totalSteps + 1))

    if args.from_step is not None:
        return list(range(args.from_step, totalSteps + 1))

    steps = set()
    for part in args.step.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            steps.update(range(int(a), int(b) + 1))
        else:
            steps.add(int(part))
    return sorted(steps)


def runStep(stepNum: int, label: str, fn) -> None:
    """Ejecuta un paso con logging de tiempo."""
    print(f"\n{'='*60}")
    print(f"PASO {stepNum}: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f"\n✓ Paso {stepNum} completado en {elapsed:.1f}s")


def main() -> None:
    args = parseArgs()
    steps = resolveSteps(args)
    print(f"Pasos a ejecutar: {steps}")

    stepDefs = {
        1: ("Perfilado inicial",       lambda: __import__("perfilado").runProfiling()),
        2: ("Limpieza intra-fuente",   lambda: __import__("limpieza").runCleaning()),
        3: ("Fusión y Golden Record",  lambda: __import__("fusion").runFusion()),
        4: ("Perfilado post + outliers + auditoría",
                                       lambda: __import__("analisis_calidad").runAnalysis()),
        5: ("Análisis, modelos y conclusiones",
                                       lambda: __import__("analisis").runAnalysis()),
    }

    tTotal = time.time()
    for s in steps:
        if s not in stepDefs:
            print(f"Paso {s} desconocido, se omite.")
            continue
        label, fn = stepDefs[s]
        try:
            runStep(s, label, fn)
        except Exception as e:
            print(f"\n✗ ERROR en paso {s} ({label}): {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Pipeline completado en {time.time() - tTotal:.1f}s")
    print(f"Resultados en:  outputs/")
    print(f"Gráficas en:    figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
