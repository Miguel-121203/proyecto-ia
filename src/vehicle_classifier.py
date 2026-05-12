"""Modelo 1 — Clasificador de tipo de vehículo (moto / carro).

Arquitectura: MobileNetV2 + transfer learning.
Input: imagen RGB 224x224, valores en [0, 1].
Output: etiqueta + confianza.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import INFERENCE

logger = logging.getLogger(__name__)

# Orden alfabético por flow_from_directory: "Bike" (idx 0) = moto, "Car" (idx 1) = carro.
CLASSES: tuple[str, str] = ("moto", "carro")


class VehicleClassifier:
    """Wrapper de inferencia para el clasificador de vehículos."""

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self.model = self._load(weights_path)

    @staticmethod
    def _load(weights_path: Path):
        if not weights_path.exists():
            logger.warning(
                "Pesos del clasificador de vehículo no encontrados en %s. "
                "El pipeline levantará NotImplementedError hasta que se entrenen.",
                weights_path,
            )
            return None
        from tensorflow.keras.models import load_model  # import diferido

        logger.info("Cargando vehicle_classifier desde %s", weights_path)
        return load_model(weights_path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Clasifica una imagen como moto o carro.

        Args:
            image: Array RGB de shape (H, W, 3), dtype uint8 o float32 en [0, 1].

        Returns:
            Tupla (etiqueta, confianza).
        """
        if self.model is None:
            raise NotImplementedError("vehicle_classifier no entrenado todavía.")

        tensor = self._preprocess(image)
        probs = self.model.predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        from tensorflow.image import resize  # import diferido

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        resized = resize(image, INFERENCE.vehicle_input_size).numpy()
        return np.expand_dims(resized, axis=0)
