"""
Motor de reconocimiento facial.

Encapsula el estado necesario para reconocer (rostros conocidos,
clasificador entrenado, umbrales) detrás de un único método
recognize(), separado de la interfaz gráfica para que se pueda probar
o reutilizar de forma independiente (por ejemplo, en un script de
evaluación por lote más adelante).
"""

import numpy as np
import torch
from numpy.linalg import norm

from config import CLASSIFIER_CONFIDENCE_THRESHOLD, CLASSIFIER_COSINE_GATE
from geometry import extract_geometric_features


class RecognitionEngine:
    def __init__(self):
        # known_faces: nombre -> lista de muestras, cada una
        # {"embedding": np.array(512), "geom": np.array(10) o None}
        self.known_faces = {}

        # Clasificador entrenado (None hasta que se entrene o cargue)
        self.classifier = None
        self.class_labels = []
        self.feature_mean = None
        self.feature_std = None

        # Umbral de similitud de coseno (respaldo, y para verificación
        # 1 a 1 contra una persona específica)
        self.threshold = 0.65

    # ------------------------------------------------------------------
    # Comparación por similitud de coseno
    # ------------------------------------------------------------------
    def recognize_cosine(self, embedding, only_person=None):
        best_name = None
        best_score = -1.0

        items = self.known_faces.items()
        if only_person is not None:
            if only_person not in self.known_faces:
                return None, -1.0
            items = [(only_person, self.known_faces[only_person])]

        for name, samples in items:
            for sample in samples:
                known_embedding = sample["embedding"]
                similarity = np.dot(embedding, known_embedding) / (
                    norm(embedding) * norm(known_embedding)
                )
                if similarity > best_score:
                    best_score = similarity
                    best_name = name

        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score

    # ------------------------------------------------------------------
    # Clasificador entrenado (red neuronal)
    # ------------------------------------------------------------------
    def predict_with_classifier(self, embedding, geom, only_person=None):
        """Usa la red neuronal ya entrenada para predecir la
        identidad. Devuelve (nombre_o_None, confianza).

        - only_person=None (identificación 1:N / modo automático):
          se elige la clase con mayor probabilidad (argmax) entre
          TODAS las personas que el clasificador conoce.
        - only_person="Nombre" (verificación 1:1): en vez de competir
          contra todas las clases, se lee directamente la
          probabilidad que el clasificador le asignó a ESA clase en
          particular. Así "confianza" significa lo mismo en los dos
          modos -- qué tan seguro está el clasificador de que la cara
          es esa persona -- y los porcentajes que ve el usuario dejan
          de venir de dos métricas distintas (softmax vs. coseno) según
          qué opción haya elegido en la interfaz.

        Si only_person no es una clase que el clasificador conozca
        (por ejemplo, se registró después del último entrenamiento),
        devuelve (None, 0.0) -- el llamador (recognize()) debe usar
        recognize_cosine() como respaldo en ese caso.

        Las características se normalizan con la misma media/
        desviación estándar calculadas durante el entrenamiento. Sin
        esto, el embedding (valores pequeños, ~0.04) y la geometría
        (valores ~0.3-3.0) tendrían escalas muy distintas, y la red
        terminaría dándole casi todo el peso a la geometría -- más
        ruidosa y menos confiable con pocas muestras -- ignorando el
        embedding, que es la parte robusta ya entrenada por ArcFace.
        """
        if self.classifier is None or geom is None:
            return None, 0.0

        if only_person is not None and only_person not in self.class_labels:
            return None, 0.0

        combined = np.concatenate([embedding, geom]).astype(np.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            combined = (combined - self.feature_mean) / self.feature_std

        x = torch.tensor(combined).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)[0]

        if only_person is None:
            conf, idx = torch.max(probs, dim=0)
            if conf.item() >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                return self.class_labels[idx.item()], conf.item()
            return None, conf.item()

        idx = self.class_labels.index(only_person)
        conf = probs[idx].item()
        if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD:
            return only_person, conf
        return None, conf

    # ------------------------------------------------------------------
    # Punto único de reconocimiento
    # ------------------------------------------------------------------
    def recognize(self, face, only_person=None):
        """Punto único de reconocimiento, usado tanto en modo
        automático (identificación 1:N) como en verificación 1 a 1
        contra una persona específica.

        Si hay un clasificador entrenado Y la persona relevante (todas,
        en modo automático; o la seleccionada, en verificación 1 a 1)
        es una clase que el clasificador conoce: se usa el
        clasificador como primer filtro, PERO la predicción solo se
        acepta si además la similitud de coseno contra las muestras
        reales de esa persona (embedding de ArcFace, no la geometría)
        supera CLASSIFIER_COSINE_GATE. Esta doble verificación evita
        que el clasificador -- entrenado con pocas muestras, por lo
        que no aprendió a "rechazar" caras desconocidas -- etiquete
        con confianza a alguien que en realidad no está registrado, o
        confunda a una persona con otra.

        Usar el mismo clasificador en ambos modos hace que la
        "confianza" mostrada signifique lo mismo sin importar qué
        opción se haya elegido en la interfaz -- antes, verificación 1
        a 1 usaba solo similitud de coseno (típicamente 60-85% incluso
        en aciertos claros) mientras que automático usaba la confianza
        del clasificador (típicamente cercana a 0% o 100%), y ambos
        números se mostraban igual como "%", lo cual parecía una
        inconsistencia del sistema sin serlo.

        Si no hay clasificador entrenado, o la persona pedida no es
        una de sus clases (por ejemplo, se registró después del
        último entrenamiento), se usa directamente la comparación por
        similitud de coseno como respaldo.
        """
        embedding = face.embedding
        geom = extract_geometric_features(face)

        classifier_usable = self.classifier is not None and (
            only_person is None or only_person in self.class_labels
        )

        if classifier_usable:
            candidate_name, confidence = self.predict_with_classifier(embedding, geom, only_person=only_person)
            if candidate_name is None:
                return None, confidence

            _, cosine_score = self.recognize_cosine(embedding, only_person=candidate_name)
            if cosine_score >= CLASSIFIER_COSINE_GATE:
                return candidate_name, confidence
            return None, confidence

        return self.recognize_cosine(embedding, only_person=only_person)
