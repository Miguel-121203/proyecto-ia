"""Entrada unificada: entrena modelos que falten y luego sirve la app Flask.

Uso:
    python run.py              # modo completo (mejor accuracy, ~1 h primera vez)
    python run.py --fast       # modo demo (entrena pocas imgs, ~3-5 min)

Cada modelo se entrena con un script `scripts/train_*.py` dedicado.
Los scripts streamean su progreso en vivo al stdout.
Si los pesos ya existen en `models/`, se saltan y se arranca Flask directamente.

Para Docker / HF Spaces se usa `python app/app.py` (los modelos ya vienen en la imagen).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SCRIPTS_DIR = ROOT / "scripts"
TRAINING_DIR = SCRIPTS_DIR / "training"
DATA_SCRIPTS_DIR = SCRIPTS_DIR / "data"

# Orden de entrenamiento: (peso esperado, script que lo produce dentro de scripts/training/)
TRAINING_PLAN: list[tuple[str, str]] = [
    ("vehicle_classifier.h5", "train_vehicle.py"),
    ("plate_detector.pt", "train_yolo.py"),
    ("origin_classifier.h5", "train_origin.py"),
]


def _run_script(script: Path, fast: bool) -> None:
    logger.info("→ Ejecutando: %s%s", script.name, " (FAST)" if fast else "")
    cmd = [sys.executable, "-u", str(script)]
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)


def _ensure_vehicle_split() -> None:
    """Genera data/processed/vehicles_split/{train,val,test} si no existe."""
    split_dir = ROOT / "data" / "processed" / "vehicles_split"
    if split_dir.exists() and all((split_dir / s).exists() for s in ("train", "val", "test")):
        return
    logger.info("→ Generando split 70/15/15 del dataset de vehículos...")
    subprocess.run([sys.executable, str(DATA_SCRIPTS_DIR / "split_vehicles.py")], check=True)


def train_missing_models(fast: bool) -> None:
    missing = [(w, s) for w, s in TRAINING_PLAN if not (MODELS_DIR / w).exists()]
    if not missing:
        logger.info("✓ Todos los modelos ya están entrenados. Saltando entrenamiento.")
        return

    logger.warning(
        "Faltan %d modelo(s): %s. Entrenando%s.",
        len(missing),
        ", ".join(w for w, _ in missing),
        " en modo FAST" if fast else "",
    )
    for weight, script in missing:
        if script == "train_vehicle.py":
            _ensure_vehicle_split()
        _run_script(TRAINING_DIR / script, fast=fast)
        if not (MODELS_DIR / weight).exists():
            raise RuntimeError(
                f"El script {script} terminó pero no produjo {weight}. Revisar logs."
            )
        logger.info("✓ %s entrenado y guardado.", weight)


def serve() -> None:
    logger.info("Arrancando Flask app en http://localhost:7860")
    sys.path.insert(0, str(ROOT))
    from app.app import app, _warm_up  # noqa: E402

    _warm_up()
    app.run(host="0.0.0.0", port=7860, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Entrenamiento rápido (demo).")
    args = parser.parse_args()

    train_missing_models(fast=args.fast)
    serve()


if __name__ == "__main__":
    main()
