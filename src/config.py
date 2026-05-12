"""Configuración central del proyecto.

Centraliza paths y hiperparámetros de inferencia. Evita hardcodear rutas
en el resto del código.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_models_dir() -> Path:
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "models").resolve()


MODELS_DIR: Path = _resolve_models_dir()


@dataclass(frozen=True)
class ModelPaths:
    """Rutas absolutas a los pesos de cada modelo."""

    vehicle_classifier: Path = MODELS_DIR / "vehicle_classifier.h5"
    plate_detector: Path = MODELS_DIR / "plate_detector.pt"
    origin_classifier: Path = MODELS_DIR / "origin_classifier.h5"
    origin_classes: Path = MODELS_DIR / "origin_classes.json"


@dataclass(frozen=True)
class InferenceConfig:
    """Hiperparámetros de inferencia del pipeline."""

    vehicle_input_size: tuple[int, int] = (224, 224)
    vehicle_threshold: float = 0.5
    yolo_input_size: int = 640
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    origin_input_size: tuple[int, int] = (300, 300)
    origin_num_classes: int = 51


MODEL_PATHS = ModelPaths()
INFERENCE = InferenceConfig()
