"""
Extracción de características geométricas a partir de los 106
landmarks faciales de insightface.

Estas medidas son proporciones estructurales (distancia entre ojos,
mandíbula, simetría, etc.) pensadas para ser más estables que el
embedding profundo ante cambios superficiales de apariencia (cabello,
barba, lentes, iluminación).
"""

import numpy as np

# Debe coincidir con la cantidad de valores devueltos por
# extract_geometric_features().
GEOM_FEATURE_DIM = 10

# Se incrementa cada vez que cambia el CÁLCULO de las características
# geométricas (por ejemplo, al corregir el bug de eye_dist sin
# normalizar). Como solo se guardan los vectores calculados -- no las
# fotos originales -- los datos guardados con una versión anterior ya
# no son comparables con los nuevos y deben recalcularse volviendo a
# capturar/importar esas muestras. face_store.load_known_faces() usa
# este valor para detectar datos desactualizados y avisar en vez de
# mezclarlos silenciosamente con datos nuevos (lo cual arruinaría el
# entrenamiento del clasificador).
GEOM_FEATURE_VERSION = 2


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def extract_geometric_features(face):
    """Calcula un vector de medidas geométricas (ojos, nariz, boca,
    mandíbula, simetría) a partir de los 106 landmarks faciales.

    Todas las medidas son proporciones normalizadas por la distancia
    entre ojos, por lo que NO dependen de qué tan cerca esté la
    persona de la cámara ni del tamaño de la imagen. Al ser medidas
    estructurales/óseas, deberían verse poco afectadas por cambios de
    apariencia como cabello, barba o lentes.

    ADVERTENCIA: los rangos de índices de landmarks usados aquí son
    los estándar documentados para insightface (106 puntos), pero
    pueden variar levemente según versión. Verifica con
    verificar_landmarks.py antes de confiar plenamente en esto.
    """
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return None

    jaw = lm[0:33]
    nose_bottom = lm[55:66]
    right_eye = lm[66:76]
    left_eye = lm[76:86]
    mouth = lm[86:104]

    right_eye_center = right_eye.mean(axis=0)
    left_eye_center = left_eye.mean(axis=0)
    nose_tip = nose_bottom.mean(axis=0)
    mouth_center = mouth.mean(axis=0)
    jaw_left = jaw[0]
    jaw_right = jaw[-1]
    chin = jaw[16]

    eye_dist = _dist(left_eye_center, right_eye_center)
    if eye_dist < 1e-6:
        eye_dist = 1.0

    eye_nose_right = _dist(right_eye_center, nose_tip) / eye_dist
    eye_nose_left = _dist(left_eye_center, nose_tip) / eye_dist
    eye_chin_right = _dist(right_eye_center, chin) / eye_dist
    eye_chin_left = _dist(left_eye_center, chin) / eye_dist
    jaw_width_norm = _dist(jaw_left, jaw_right) / eye_dist
    face_height_norm = (eye_chin_right + eye_chin_left) / 2.0
    if face_height_norm < 1e-6:
        face_height_norm = 1.0

    features = [
        # IMPORTANTE: aquí NO se incluye eye_dist en píxeles crudos.
        # Antes se incluía sin normalizar, lo cual hacía que el
        # reconocimiento dependiera de qué tan cerca estaba la persona
        # de la cámara (algo que no tiene relación con su identidad).
        # En su lugar, se agrega la proporción ancho/alto del rostro,
        # que sí es invariante a la escala y a la distancia.
        _dist(nose_tip, mouth_center) / eye_dist,
        jaw_width_norm,                                # ancho de mandíbula
        _dist(mouth[0], mouth[9]) / eye_dist,          # ancho de boca
        eye_nose_right,
        eye_nose_left,
        eye_chin_right,
        eye_chin_left,
        abs(eye_nose_right - eye_nose_left),           # simetría ojo-nariz
        abs(eye_chin_right - eye_chin_left),           # simetría ojo-mentón
        jaw_width_norm / face_height_norm,             # proporción ancho/alto del rostro
    ]
    return np.array(features, dtype=np.float32)


# Debe coincidir con la cantidad de valores devueltos por
# extract_upper_face_features().
GEOM_UPPER_FEATURE_DIM = 6


def extract_upper_face_features(face):
    """Calcula un subconjunto de características geométricas usando
    SOLO landmarks de la mitad SUPERIOR del rostro (cejas y ojos) --
    la zona que una mascarilla prácticamente nunca cubre.

    Es un complemento de extract_geometric_features(), no un
    reemplazo: ese vector completo usa también nariz, boca y
    mandíbula, que con mascarilla puesta insightface igual "calcula"
    pero sobre landmarks estimados/interpolados (no observados
    realmente), así que dejan de ser confiables. Este vector reducido
    sirve como referencia de qué tan reconocible sigue siendo alguien
    usando únicamente la parte de la cara que casi siempre permanece
    visible.

    Al ser un vector más corto (menos zonas del rostro aportan
    información), tiene menos poder discriminativo que el vector
    completo -- está pensado para análisis de robustez y como
    posible respaldo ante oclusión, no para reemplazar el
    reconocimiento normal.
    """
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return None

    right_eyebrow = lm[33:42]
    left_eyebrow = lm[42:51]
    right_eye = lm[66:76]
    left_eye = lm[76:86]

    right_eye_center = right_eye.mean(axis=0)
    left_eye_center = left_eye.mean(axis=0)
    right_eyebrow_center = right_eyebrow.mean(axis=0)
    left_eyebrow_center = left_eyebrow.mean(axis=0)

    eye_dist = _dist(left_eye_center, right_eye_center)
    if eye_dist < 1e-6:
        eye_dist = 1.0

    right_eye_width = _dist(right_eye[0], right_eye[len(right_eye) // 2])
    left_eye_width = _dist(left_eye[0], left_eye[len(left_eye) // 2])
    eyebrow_eye_right = _dist(right_eyebrow_center, right_eye_center) / eye_dist
    eyebrow_eye_left = _dist(left_eyebrow_center, left_eye_center) / eye_dist

    features = [
        eyebrow_eye_right,                                     # distancia ceja-ojo (derecha)
        eyebrow_eye_left,                                      # distancia ceja-ojo (izquierda)
        _dist(right_eyebrow[0], left_eyebrow[-1]) / eye_dist,  # ancho entre extremos de cejas
        right_eye_width / eye_dist,                            # ancho del ojo derecho
        left_eye_width / eye_dist,                             # ancho del ojo izquierdo
        abs(eyebrow_eye_right - eyebrow_eye_left),             # simetría ceja-ojo
    ]
    return np.array(features, dtype=np.float32)
