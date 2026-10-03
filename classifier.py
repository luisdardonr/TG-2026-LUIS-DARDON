"""
Red neuronal clasificadora (PyTorch) y funciones de entrenamiento,
carga y guardado del modelo.

Esta es la única parte del sistema que se entrena de verdad con
backpropagation (función de pérdida, épocas, optimizador) sobre las
muestras registradas -- es lo que se reentrena cuando se agregan
nuevas variaciones de apariencia.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    CLASSIFIER_LABELS_PATH,
    CLASSIFIER_MODEL_PATH,
    TRAIN_EPOCHS,
    TRAIN_LR,
)


class FaceClassifier(nn.Module):
    """Red neuronal feedforward que recibe [embedding + geometría] y
    predice a cuál de las personas registradas corresponde."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier_from_samples(known_faces, min_samples_per_person):
    """Entrena el clasificador a partir del diccionario known_faces
    ({nombre: [{"embedding":..., "geom":...}, ...]}).

    Guarda el modelo y sus metadatos en disco y devuelve un dict con:
        model, class_labels, feature_mean, feature_std,
        train_acc, final_loss, n_samples, personas_insuficientes

    Lanza ValueError con un mensaje listo para mostrar al usuario si
    no hay datos suficientes para entrenar.
    """
    names = list(known_faces.keys())
    if len(names) < 2:
        raise ValueError(
            "Se necesitan al menos 2 personas registradas (con capturas) "
            "para entrenar el clasificador."
        )

    # Conteo de muestras válidas (con geometría) por persona. Se exige
    # un mínimo por persona porque con 1-2 capturas la red no tiene
    # forma de aprender rasgos que generalicen: memoriza esas fotos
    # exactas y falla apenas cambia el ángulo, la luz o la distancia a
    # la cámara.
    counts = {name: sum(1 for s in known_faces[name] if s["geom"] is not None) for name in names}
    personas_insuficientes = {n: c for n, c in counts.items() if c < min_samples_per_person}

    X, y = [], []
    for idx, name in enumerate(names):
        for sample in known_faces[name]:
            geom = sample["geom"]
            if geom is None:
                continue  # muestras sin geometría (formato viejo/desactualizado) no entran al entrenamiento
            combined = np.concatenate([sample["embedding"], geom])
            X.append(combined)
            y.append(idx)

    if len(X) < len(names) * 2:
        raise ValueError(
            "Muy pocas muestras con datos geométricos válidos.\n"
            "Registra o importa más capturas (con landmarks) antes de entrenar."
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Normalización: el embedding (valores ~0.04) y la geometría
    # (valores ~0.3-3.0) tienen escalas muy distintas. Sin esto, la
    # red termina dándole casi todo el peso a la geometría --más
    # ruidosa y menos confiable con pocas muestras-- e ignorando el
    # embedding, que es la parte robusta ya entrenada por ArcFace.
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0  # evitar división por cero en columnas constantes
    X = (X - feature_mean) / feature_std

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    input_dim = X_tensor.shape[1]
    num_classes = len(names)

    model = FaceClassifier(input_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss = None
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1)
        train_acc = (preds == y_tensor).float().mean().item() * 100

    torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)
    with open(CLASSIFIER_LABELS_PATH, "w") as f:
        json.dump({
            "names": names,
            "input_dim": input_dim,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
        }, f)

    return {
        "model": model,
        "class_labels": names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_acc": train_acc,
        "final_loss": loss.item(),
        "n_samples": len(X),
        "personas_insuficientes": personas_insuficientes,
    }


def load_classifier():
    """Carga el clasificador guardado en disco, si existe y es
    compatible con el formato actual (con normalización).

    Devuelve (model, class_labels, feature_mean, feature_std, warning).
    warning es None si todo salió bien, o un texto explicando por qué
    se descartó/no se pudo cargar el modelo guardado.
    """
    if not (os.path.exists(CLASSIFIER_MODEL_PATH) and os.path.exists(CLASSIFIER_LABELS_PATH)):
        return None, [], None, None, None

    try:
        with open(CLASSIFIER_LABELS_PATH, "r") as f:
            meta = json.load(f)
        names = meta["names"]
        input_dim = meta["input_dim"]
        model = FaceClassifier(input_dim, len(names))
        model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))
        model.eval()

        if "feature_mean" not in meta or "feature_std" not in meta:
            # Modelo entrenado antes de agregar normalización: ya no
            # es coherente con predict_with_classifier. Se descarta y
            # se pide reentrenar.
            return None, [], None, None, (
                "El modelo guardado es de una versión anterior (sin "
                "normalización de características). Por favor vuelve "
                "a entrenar el modelo."
            )

        feature_mean = np.array(meta["feature_mean"], dtype=np.float32)
        feature_std = np.array(meta["feature_std"], dtype=np.float32)
        return model, names, feature_mean, feature_std, None
    except Exception as e:
        return None, [], None, None, f"Error cargando clasificador: {e}"
