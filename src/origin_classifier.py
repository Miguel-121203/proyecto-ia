"""Modelo 3 — Clasificador de estado/territorio USA a partir de la placa recortada.

Arquitectura: MobileNetV2 + transfer learning.
Input: placa recortada RGB redimensionada a 300×300.
Output: nombre del estado/territorio (ej. "CALIFORNIA") + confianza.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import INFERENCE, MODEL_PATHS

logger = logging.getLogger(__name__)


def _load_model(weights_path: Path):
    if not weights_path.exists():
        logger.warning("Pesos del clasificador de origen no encontrados en %s.", weights_path)
        return None

    import tensorflow as tf

    logger.info("Cargando origin_classifier (MobileNetV2) desde %s", weights_path)
    return tf.keras.models.load_model(str(weights_path), compile=False)


class OriginClassifier:
    """Wrapper de inferencia para el clasificador de estado USA."""

    def __init__(self, weights_path: Path, classes_path: Path) -> None:
        self.classes: list[str] = self._load_classes(classes_path)
        self.model = _load_model(weights_path)

    @staticmethod
    def _load_classes(classes_path: Path) -> list[str]:
        if not classes_path.exists():
            logger.warning("Archivo de clases no encontrado en %s.", classes_path)
            return []
        return json.loads(classes_path.read_text())

    def predict(self, plate_image: np.ndarray) -> tuple[str, float]:
        """Clasifica el estado USA de una placa recortada.

        Args:
            plate_image: Array RGB de shape (H, W, 3), dtype uint8.

        Returns:
            Tupla (nombre_estado, confianza).
        """
        if self.model is None or not self.classes:
            raise NotImplementedError("origin_classifier no disponible.")

        tensor = self._preprocess(plate_image)
        probs = self.model.predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        h, w = INFERENCE.origin_input_size
        resized = tf.image.resize(image, (h, w)).numpy()
        preprocessed = preprocess_input(resized.astype(np.float32))
        return np.expand_dims(preprocessed, axis=0)
