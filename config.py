"""
Configuración global del sistema de reconocimiento facial.

Centraliza rutas de persistencia, umbrales y parámetros de
entrenamiento para que todos los módulos (gui_app, classifier,
face_store, recognition_engine, metrics) usen exactamente los mismos
valores en vez de tener constantes repetidas y potencialmente
inconsistentes en cada archivo.
"""

AUTO_LABEL = "Automático (todas las personas)"

# ------------------------------------------------------------------
# Rutas de persistencia
# ------------------------------------------------------------------
METRICS_LOG_PATH = "metrics_log.csv"
KNOWN_FACES_PATH = "known_faces.json"
CLASSIFIER_MODEL_PATH = "classifier_model.pt"
CLASSIFIER_LABELS_PATH = "classifier_labels.json"

# Carpeta que se importa automáticamente al iniciar, si existe y no
# hay personas registradas todavía (estructura: dataset/nombre/*.jpg)
DATASET_PATH = "dataset"

# ------------------------------------------------------------------
# Muestreo de video para importación masiva
# ------------------------------------------------------------------
VIDEO_SAMPLE_INTERVAL = 15
VIDEO_MAX_SAMPLES = 30

# ------------------------------------------------------------------
# Evaluación en vivo
# ------------------------------------------------------------------
EVAL_LOG_INTERVAL = 15

# ------------------------------------------------------------------
# Clasificador (red neuronal entrenable)
# ------------------------------------------------------------------
# Umbral de confianza del clasificador (softmax) para aceptar una
# predicción; por debajo de esto se considera "Desconocido"
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.6

TRAIN_EPOCHS = 150
TRAIN_LR = 0.001

# Con muy pocas muestras por persona, la red neuronal no tiene manera
# de aprender rasgos que generalicen: simplemente memoriza esas fotos
# exactas y falla con cualquier variación (ángulo, luz, distancia a la
# cámara). Se recomienda un mínimo razonable antes de entrenar.
MIN_SAMPLES_PER_PERSON = 3

# Además del clasificador, se exige que la predicción quede respaldada
# por una similitud de coseno mínima contra las muestras reales de esa
# persona (ver recognition_engine.py). Esto evita que la red -- entrenada
# con pocos datos y por lo tanto poco confiable para "rechazar" rostros
# desconocidos -- etiquete con confianza a alguien que en realidad no
# está registrado.
CLASSIFIER_COSINE_GATE = 0.45

# Umbral de similitud de coseno por defecto (respaldo cuando no hay
# clasificador entrenado, y para verificación 1 a 1).
DEFAULT_COSINE_THRESHOLD = 0.65
