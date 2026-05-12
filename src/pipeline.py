"""Pipeline en cascada: vehículo → placa → clasificador de estado USA.

Uso:
    pipeline = VehiclePipeline(models_dir=Path("./models"))
    result = pipeline.predict(image_array)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.config import MODEL_PATHS, MODELS_DIR
from src.origin_classifier import OriginClassifier
from src.plate_detector import PlateDetector
from src.vehicle_classifier import VehicleClassifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Resultado de una ejecución del pipeline en cascada."""

    vehicle_type: str
    vehicle_confidence: float
    plate_bbox: tuple[int, int, int, int] | None = None
    plate_confidence: float | None = None
    origin: str | None = None
    origin_confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class VehiclePipeline:
    """Orquestador de inferencia. Carga los modelos una única vez."""

    def __init__(self, models_dir: Path | None = None) -> None:
        logger.info("Inicializando VehiclePipeline con models_dir=%s", models_dir or MODELS_DIR)

        self.vehicle = VehicleClassifier(MODEL_PATHS.vehicle_classifier)
        self.detector = PlateDetector(MODEL_PATHS.plate_detector)
        self.origin = OriginClassifier(MODEL_PATHS.origin_classifier, MODEL_PATHS.origin_classes)

    def predict(self, image: np.ndarray) -> PipelineResult:
        """Ejecuta el pipeline completo sobre una imagen.

        Args:
            image: Array RGB de shape (H, W, 3), dtype uint8.

        Returns:
            PipelineResult con los campos rellenados según qué etapas tuvieron éxito.
        """
        vehicle_label, vehicle_conf = self.vehicle.predict(image)
        result = PipelineResult(
            vehicle_type=vehicle_label,
            vehicle_confidence=vehicle_conf,
        )

        detection = self.detector.predict(image)
        if detection is None:
            logger.info("No se detectó placa en la imagen.")
            return result

        bbox, det_conf = detection
        result.plate_bbox = bbox
        result.plate_confidence = det_conf

        plate_crop = self.detector.crop(image, bbox)
        try:
            state, origin_conf = self.origin.predict(plate_crop)
            result.origin = state
            result.origin_confidence = origin_conf
        except NotImplementedError:
            logger.warning("origin_classifier aún no disponible.")

        return result
