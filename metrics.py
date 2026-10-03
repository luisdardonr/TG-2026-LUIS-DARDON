"""
Registro de métricas de evaluación a un archivo CSV, para su análisis
posterior en el informe de tesis.

Cada fila indica de dónde vino esa evaluación:
    - "vivo"    -> comparación en tiempo real frente a la cámara
    - "dataset" -> recorrido por lote sobre fotos ya existentes en disco

Guardar ambas fuentes en el mismo archivo (en vez de dos separados)
permite filtrar/agrupar por columna "fuente" al analizar los
resultados para el informe, sin perder la trazabilidad de cada tipo
de evaluación.
"""

import csv
import os
import shutil
from datetime import datetime

from config import METRICS_LOG_PATH

_HEADER = ["timestamp", "fuente", "persona_real", "prediccion", "score", "acierto"]


def ensure_metrics_file():
    """Crea metrics_log.csv si no existe. Si ya existe pero es de una
    versión anterior (sin columna 'fuente'), lo migra agregando
    fuente='vivo' a las filas existentes -- así no se pierden
    evaluaciones ya hechas. Se guarda una copia de respaldo antes de
    modificarlo."""
    if not os.path.exists(METRICS_LOG_PATH):
        _write_header()
        return

    with open(METRICS_LOG_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    if header == _HEADER:
        return

    backup_path = METRICS_LOG_PATH + ".bak"
    if not os.path.exists(backup_path):
        shutil.copyfile(METRICS_LOG_PATH, backup_path)

    migrated_rows = []
    for row in rows:
        if len(row) == 5:
            timestamp, persona_real, prediccion, score, acierto = row
            migrated_rows.append([timestamp, "vivo", persona_real, prediccion, score, acierto])
        elif len(row) == len(_HEADER):
            migrated_rows.append(row)
        # filas con un formato irreconocible se descartan silenciosamente

    _write_header()
    with open(METRICS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(migrated_rows)


def _write_header():
    with open(METRICS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)


def log_metric(ground_truth, predicted, score, correct, source="vivo"):
    with open(METRICS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            source,
            ground_truth,
            predicted or "Desconocido",
            f"{score:.4f}",
            correct,
        ])
