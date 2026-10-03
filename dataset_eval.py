"""
Evaluación por lote sobre un dataset de imágenes ya existente ("la
función estática"), como complemento a la evaluación en vivo frente a
la cámara (gui_app.toggle_evaluation / update_video).

Recorre una carpeta con estructura dataset/persona/*.jpg -- la misma
que usa face_store.import_dataset_from_path -- reconoce cada imagen
con el motor de reconocimiento ya entrenado (RecognitionEngine) y
compara el resultado contra el nombre de la carpeta, asumido como la
identidad real.

Además de acierto/fallo (la vista "clasificador"), este módulo calcula
un segundo grupo de métricas pensadas para estudiar el comportamiento
de la RED y el ALGORITMO en sí -- la vista "deep learning" que se
pidió agregar:

  - Pérdida (cross-entropy) de la red sobre la clase real de cada
    imagen, no solo si acertó el argmax. Dos personas pueden tener la
    misma precisión pero pérdidas muy distintas -- eso indica qué tan
    bien calibradas están las probabilidades del clasificador.
  - Tiempo de inferencia por imagen (detección + reconocimiento).
  - Curva de error acumulado a lo largo del recorrido (para ver si el
    sistema falla más al principio, al final, o de forma pareja).
  - Confianza promedio en aciertos vs. en fallos -- si el modelo está
    bien calibrado, debería estar más seguro cuando acierta.
  - Análisis de robustez ante oclusión (mascarilla): para cada
    persona, se arma un "perfil superior" (promedio de sus
    características de cejas/ojos dentro de este mismo dataset, ver
    geometry.extract_upper_face_features) y se compara cada imagen
    contra el perfil de su propia persona. Un fallo del reconocimiento
    principal con una similitud superior ALTA sugiere que la mitad
    superior del rostro seguía siendo reconocible -- justo el caso de
    uso de una mascarilla.
  - Detección de FUGA DE DATOS: si una imagen que se está evaluando
    es (casi) idéntica a una de las capturas ya guardadas en
    known_faces.json para esa misma persona, es señal de que esa foto
    también se usó para entrenar/registrar -- entonces no está
    midiendo qué tan bien generaliza el sistema, solo si "recuerda"
    la foto exacta. El resumen reporta cuántas imágenes sospechosas
    de fuga hay, y calcula la precisión también EXCLUYÉNDOLAS, para
    poder comparar ambos números.

Cada resultado también se registra en metrics_log.csv con
fuente="dataset" (igual que antes), y además se genera un CSV
detallado por imagen (ver _write_detailed_report) para que las
métricas se puedan graficar libremente en Excel/Python para el
informe, sin depender de que esta app tenga gráficas propias.
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from geometry import extract_geometric_features, extract_upper_face_features
from metrics import log_metric

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Dos embeddings ArcFace de la MISMA foto, procesados de nuevo por el
# mismo modelo, salen prácticamente idénticos (similitud de coseno
# >0.999) -- muchísimo más alto que dos fotos distintas de la misma
# persona real (que normalmente caen en 0.6-0.9). Por eso este umbral
# tan estricto sirve como huella digital para detectar "es la misma
# imagen" en vez de solo "es la misma persona".
LEAKAGE_SIMILARITY_THRESHOLD = 0.999


def _max_similarity_against_registered(embedding, known_samples):
    """Similitud de coseno más alta entre `embedding` y las muestras
    YA REGISTRADAS (known_faces) de una persona. Se usa para detectar
    si la imagen que se está evaluando es, en realidad, una de las
    fotos con las que ya se entrenó/registró a esa persona."""
    best = -1.0
    for sample in known_samples:
        known_embedding = sample["embedding"]
        denom = float(np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
        if denom < 1e-9:
            continue
        sim = float(np.dot(embedding, known_embedding) / denom)
        if sim > best:
            best = sim
    return best


def _cosine(a, b):
    """Similitud de coseno entre dos vectores genéricos (no
    necesariamente embeddings de 512 valores -- aquí se usa también
    sobre vectores geométricos de 6 valores)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_std(values):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(values)}


def _write_detailed_report(rows, folder):
    """Escribe un CSV con una fila por imagen evaluada -- pensado para
    que se pueda abrir en Excel y graficar (histogramas de confianza,
    curva de error, dispersión pérdida vs. tiempo, etc.) directamente
    para el informe, sin tener que construir esas gráficas dentro de
    la app."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"dataset_eval_report_{timestamp}.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imagen", "persona_real", "prediccion", "acierto", "score",
            "tiempo_inferencia_ms", "loss", "similitud_superior", "posible_fuga",
        ])
        writer.writerows(rows)
    return report_path


def evaluate_dataset(engine, folder, face_app, only_registered=True):
    """Recorre folder/persona/*.jpg, reconoce cada imagen y la compara
    contra el nombre de la carpeta (identidad real).

    only_registered: si True (por defecto), solo evalúa carpetas cuyo
    nombre ya está en engine.known_faces. Esto evita que carpetas de
    personas que nunca se registraron (o que se nombraron distinto en
    el dataset) se cuenten como errores del sistema, cuando en
    realidad el sistema nunca tuvo forma de reconocerlas.

    Devuelve un dict con, entre otras claves:
        total, correct, accuracy, sin_rostro, carpetas_omitidas,
        per_person, avg_inference_ms, max_inference_ms, avg_loss,
        confidence_correct, confidence_incorrect, error_curve,
        upper_face_analysis, posible_fuga_count, imagenes_con_fuga,
        total_sin_fuga, correct_sin_fuga, accuracy_sin_fuga, report_path
    (ver el cuerpo de la función para el detalle de cada una).
    """
    root_path = Path(folder)
    summary = {
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "sin_rostro": 0,
        "carpetas_omitidas": [],
        "per_person": {},

        # -- Métricas estilo deep learning (más allá de acierto/fallo) --
        "avg_inference_ms": None,
        "max_inference_ms": None,
        "avg_loss": None,  # cross-entropy promedio; None si no hay clasificador entrenado
        "confidence_correct": {"mean": None, "std": None, "n": 0},
        "confidence_incorrect": {"mean": None, "std": None, "n": 0},
        "error_curve": [],  # [(n_muestras_procesadas, tasa_error_acumulada_%), ...]

        # -- Análisis de robustez ante oclusión (mascarilla) --
        "upper_face_analysis": {
            "n_personas_con_perfil": 0,
            "similitud_superior_aciertos": {"mean": None, "std": None, "n": 0},
            "similitud_superior_fallos": {"mean": None, "std": None, "n": 0},
        },

        # -- Detección de fuga de datos (evaluar con fotos de entrenamiento) --
        "posible_fuga_count": 0,
        "imagenes_con_fuga": [],  # ["persona/archivo.jpg", ...]
        "total_sin_fuga": 0,
        "correct_sin_fuga": 0,
        "accuracy_sin_fuga": None,  # None si no hubo forma de calcularla (ver más abajo)

        "report_path": None,
    }
    if not root_path.exists():
        return summary

    # ------------------------------------------------------------------
    # Primera pasada: detectar cada rostro UNA sola vez y calcular sus
    # características superiores (cejas+ojos). Se hace en dos pasadas
    # porque el "perfil superior" de cada persona se calcula como el
    # promedio de SUS propias imágenes dentro de este dataset, así que
    # hace falta tenerlas todas antes de poder comparar cada imagen
    # contra ese perfil.
    # ------------------------------------------------------------------
    records = []
    for person_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        ground_truth = person_dir.name
        if only_registered and ground_truth not in engine.known_faces:
            summary["carpetas_omitidas"].append(ground_truth)
            continue

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = face_app.get(img)
            if len(faces) == 0:
                summary["sin_rostro"] += 1
                continue

            face = faces[0]
            records.append({
                "img_name": img_path.name,
                "ground_truth": ground_truth,
                "face": face,
                "geom_upper": extract_upper_face_features(face),
            })

    if not records:
        return summary

    # Perfil superior promedio por persona (cejas+ojos), calculado
    # dentro de este mismo dataset -- no requiere haber guardado nada
    # de antemano en known_faces.json.
    upper_profiles = {}
    for name in {r["ground_truth"] for r in records}:
        vectors = [r["geom_upper"] for r in records
                   if r["ground_truth"] == name and r["geom_upper"] is not None]
        if vectors:
            upper_profiles[name] = np.mean(np.stack(vectors), axis=0)
    summary["upper_face_analysis"]["n_personas_con_perfil"] = len(upper_profiles)

    # ------------------------------------------------------------------
    # Segunda pasada: reconocer cada imagen y acumular las métricas.
    # ------------------------------------------------------------------
    inference_times = []
    losses = []
    conf_correct, conf_incorrect = [], []
    upper_sim_correct, upper_sim_incorrect = [], []
    running_errors = 0
    report_rows = []
    total_sin_fuga = 0
    correct_sin_fuga = 0

    for i, record in enumerate(records, start=1):
        ground_truth = record["ground_truth"]
        face = record["face"]
        geom_upper = record["geom_upper"]

        t0 = time.perf_counter()
        predicted, score = engine.recognize(face, only_person=None)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(elapsed_ms)

        is_correct = (predicted == ground_truth)
        summary["total"] += 1
        if is_correct:
            summary["correct"] += 1
        else:
            running_errors += 1

        stats = summary["per_person"].setdefault(
            ground_truth, {"total": 0, "correct": 0, "accuracy": 0.0}
        )
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1

        # Pérdida (cross-entropy) sobre la clase REAL, si el
        # clasificador la conoce. Esto evalúa la red como red neuronal
        # -- qué tan bien calibradas están sus probabilidades -- no
        # solo si acertó el argmax. predict_with_classifier() ya
        # devuelve la probabilidad cruda de la clase pedida sin
        # importar si supera el umbral de aceptación.
        loss_value = None
        if engine.classifier is not None and ground_truth in engine.class_labels:
            geom_full = extract_geometric_features(face)
            _, true_class_prob = engine.predict_with_classifier(
                face.embedding, geom_full, only_person=ground_truth
            )
            if true_class_prob and true_class_prob > 1e-9:
                loss_value = -math.log(true_class_prob)
                losses.append(loss_value)

        if is_correct:
            conf_correct.append(score)
        else:
            conf_incorrect.append(score)

        # Similitud superior contra el perfil de la PROPIA persona,
        # sin importar si el reconocimiento principal acertó. Un fallo
        # con similitud superior alta sugiere que la mitad superior
        # del rostro seguía siendo reconocible por sí sola.
        upper_sim = None
        if geom_upper is not None and ground_truth in upper_profiles:
            upper_sim = _cosine(geom_upper, upper_profiles[ground_truth])
            if is_correct:
                upper_sim_correct.append(upper_sim)
            else:
                upper_sim_incorrect.append(upper_sim)

        # Detección de fuga: ¿esta imagen es (casi) idéntica a una foto
        # ya registrada para esta misma persona? Si es así, evaluarla
        # no dice nada sobre qué tan bien generaliza el sistema -- es
        # la misma foto con la que se entrenó.
        leak_suspect = False
        if ground_truth in engine.known_faces:
            max_sim = _max_similarity_against_registered(
                face.embedding, engine.known_faces[ground_truth]
            )
            leak_suspect = max_sim >= LEAKAGE_SIMILARITY_THRESHOLD

        if leak_suspect:
            summary["posible_fuga_count"] += 1
            summary["imagenes_con_fuga"].append(f"{ground_truth}/{record['img_name']}")
        else:
            total_sin_fuga += 1
            if is_correct:
                correct_sin_fuga += 1

        log_metric(ground_truth, predicted, score, is_correct, source="dataset")

        error_rate = running_errors / i * 100
        summary["error_curve"].append((i, error_rate))

        report_rows.append([
            record["img_name"], ground_truth, predicted or "Desconocido", is_correct,
            f"{score:.4f}",
            f"{elapsed_ms:.2f}",
            f"{loss_value:.4f}" if loss_value is not None else "",
            f"{upper_sim:.4f}" if upper_sim is not None else "",
            leak_suspect,
        ])

    # ------------------------------------------------------------------
    # Cierre: agregados finales
    # ------------------------------------------------------------------
    for stats in summary["per_person"].values():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"] * 100

    if summary["total"] > 0:
        summary["accuracy"] = summary["correct"] / summary["total"] * 100

    if inference_times:
        summary["avg_inference_ms"] = float(np.mean(inference_times))
        summary["max_inference_ms"] = float(np.max(inference_times))

    if losses:
        summary["avg_loss"] = float(np.mean(losses))

    summary["confidence_correct"] = _mean_std(conf_correct)
    summary["confidence_incorrect"] = _mean_std(conf_incorrect)
    summary["upper_face_analysis"]["similitud_superior_aciertos"] = _mean_std(upper_sim_correct)
    summary["upper_face_analysis"]["similitud_superior_fallos"] = _mean_std(upper_sim_incorrect)

    summary["total_sin_fuga"] = total_sin_fuga
    summary["correct_sin_fuga"] = correct_sin_fuga
    if total_sin_fuga > 0:
        summary["accuracy_sin_fuga"] = correct_sin_fuga / total_sin_fuga * 100

    summary["report_path"] = _write_detailed_report(report_rows, folder)

    return summary