"""Modelo 2 — Detector de placa (YOLOv8n fine-tuned).

Input: imagen RGB.
Output: bounding box (x, y, w, h) en píxeles + confianza, o None.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import INFERENCE

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]


class PlateDetector:
    """Wrapper de inferencia para YOLOv8 fine-tuned."""

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self.model = self._load(weights_path)

    @staticmethod
    def _load(weights_path: Path):
        if not weights_path.exists():
            logger.warning(
                "Pesos de YOLO para placas no encontrados en %s.",
                weights_path,
            )
            return None
        from ultralytics import YOLO  # import diferido

        logger.info("Cargando plate_detector desde %s", weights_path)
        return YOLO(str(weights_path))

    def predict(self, image: np.ndarray) -> tuple[BBox, float] | None:
        """Detecta la placa de mayor confianza en la imagen.

        Args:
            image: Array RGB de shape (H, W, 3).

        Returns:
            ((x, y, w, h), confianza) si hay detección, None en caso contrario.
        """
        if self.model is None:
            raise NotImplementedError("plate_detector no entrenado todavía.")

        results = self.model.predict(
            image,
            imgsz=INFERENCE.yolo_input_size,
            conf=INFERENCE.yolo_confidence,
            iou=INFERENCE.yolo_iou,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        idx = int(np.argmax(boxes.conf.cpu().numpy()))
        xyxy = boxes.xyxy.cpu().numpy()[idx]
        conf = float(boxes.conf.cpu().numpy()[idx])

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        return (x1, y1, x2 - x1, y2 - y1), conf

    @staticmethod
    def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Recorta la región de la placa respetando límites de la imagen."""
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        return image[y1:y2, x1:x2]
