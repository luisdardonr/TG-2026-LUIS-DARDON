"""
Persistencia de rostros conocidos (embeddings + geometría) e
importación masiva desde carpetas de imágenes o videos.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from config import KNOWN_FACES_PATH, VIDEO_MAX_SAMPLES, VIDEO_SAMPLE_INTERVAL
from geometry import GEOM_FEATURE_VERSION, extract_geometric_features


def load_known_faces():
    """Carga known_faces.json.

    Formato:
        {
          "__meta__": {"geom_version": N},
          "nombre": [{"embedding": [...512...], "geom": [...10...] o null}, ...]
        }
    Compatible con formatos anteriores (embedding suelto o lista plana
    de embeddings sin geometría): en ese caso geom queda None.

    Si el archivo se guardó con una versión anterior del cálculo de
    características geométricas, esos vectores "geom" se descartan
    (quedan en None) para no mezclar datos incompatibles en el
    entrenamiento; los embeddings (ArcFace) sí se conservan siempre.

    Devuelve (known_faces, mensajes) donde mensajes es una lista de
    strings informativos para mostrar en el log de la interfaz.
    """
    known_faces = {}
    messages = []

    if not os.path.exists(KNOWN_FACES_PATH):
        return known_faces, messages

    try:
        with open(KNOWN_FACES_PATH, "r") as f:
            data = json.load(f)

        meta = data.pop("__meta__", {})
        saved_version = meta.get("geom_version")
        geom_desactualizada = saved_version != GEOM_FEATURE_VERSION

        for name, samples_raw in data.items():
            samples = []
            for item in samples_raw:
                if isinstance(item, dict):
                    emb = np.array(item["embedding"], dtype=np.float32)
                    geom = np.array(item["geom"], dtype=np.float32) if item.get("geom") is not None else None
                else:
                    emb = np.array(item, dtype=np.float32)
                    geom = None

                if geom_desactualizada:
                    geom = None  # descartar geometría calculada con una fórmula distinta

                samples.append({"embedding": emb, "geom": geom})
            known_faces[name] = samples

        total = sum(len(v) for v in known_faces.values())
        messages.append(f"Cargados {len(known_faces)} personas ({total} capturas totales)")

        if geom_desactualizada and total > 0:
            messages.append(
                "Aviso: las características geométricas guardadas son de una "
                "versión anterior (o no existían) y no son compatibles con los "
                "cálculos actuales -- se descartaron para evitar entrenar con "
                "datos mezclados. Los embeddings se conservaron. Para volver a "
                "tener geometría disponible, vuelve a capturar/importar las "
                "fotos de cada persona (o al menos algunas) y reentrena."
            )
    except Exception as e:
        messages.append(f"Error cargando {KNOWN_FACES_PATH}: {e}")

    return known_faces, messages


def save_known_faces(known_faces):
    data = {"__meta__": {"geom_version": GEOM_FEATURE_VERSION}}
    for name, samples in known_faces.items():
        data[name] = [
            {
                "embedding": s["embedding"].tolist(),
                "geom": s["geom"].tolist() if s["geom"] is not None else None,
            }
            for s in samples
        ]
    with open(KNOWN_FACES_PATH, "w") as f:
        json.dump(data, f)


def import_dataset_from_path(known_faces, folder, face_app):
    """Importa una carpeta con subcarpetas por persona
    (dataset/nombre/*.jpg) hacia known_faces (se modifica in place).

    Devuelve (resumen_lines, total_agregado). Guarda automáticamente
    si se agregó algo.
    """
    root_path = Path(folder)
    if not root_path.exists():
        return [], 0

    resumen = []
    total_agregado = 0

    for person_dir in root_path.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        added = 0
        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = face_app.get(img)
            if len(faces) == 0:
                continue
            face = faces[0]
            geom = extract_geometric_features(face)
            known_faces.setdefault(name, []).append({"embedding": face.embedding, "geom": geom})
            added += 1
        if added:
            resumen.append(f"   {name}: +{added} capturas")
            total_agregado += added

    if total_agregado:
        save_known_faces(known_faces)

    return resumen, total_agregado


def extract_embeddings_from_video(known_faces, filepath, name, face_app):
    """Extrae muestras de un video (una sola persona por video) hacia
    known_faces (se modifica in place). Devuelve la cantidad de
    capturas agregadas. Guarda automáticamente si se agregó algo."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0

    frame_idx = 0
    added = 0

    while added < VIDEO_MAX_SAMPLES:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % VIDEO_SAMPLE_INTERVAL == 0:
            faces = face_app.get(frame)
            if len(faces) > 0:
                face = faces[0]
                geom = extract_geometric_features(face)
                known_faces.setdefault(name, []).append({"embedding": face.embedding, "geom": geom})
                added += 1
        frame_idx += 1

    cap.release()
    if added:
        save_known_faces(known_faces)
    return added
