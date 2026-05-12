"""Entrenamiento del Modelo 2 (YOLOv8s) con progreso en vivo.

Dataset esperado en `data/raw/modelo2/`:
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt        (formato YOLO)

Si el dataset no está en YOLO sino en Pascal VOC (carpeta annotations/ con
XMLs), se convierte automáticamente con `scripts/data/voc_to_yolo.py`.

Uso desde la raíz del proyecto:
    python scripts/training/train_yolo.py            # completo (30 épocas)
    python scripts/training/train_yolo.py --fast     # demo (~6 épocas)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "raw" / "modelo2"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
MODELS_DIR = ROOT / "models"


def _is_voc_dataset(dataset_dir: Path) -> bool:
    """Detecta si el dataset está en formato Pascal VOC (carpeta annotations con XMLs)."""
    ann = dataset_dir / "annotations"
    return ann.exists() and any(ann.glob("*.xml"))


def _is_yolo_dataset(dataset_dir: Path) -> bool:
    """Detecta si el dataset ya está en formato YOLO con los 3 splits."""
    return all(
        (dataset_dir / "images" / split).exists() and any((dataset_dir / "images" / split).iterdir())
        for split in ("train", "val", "test")
    )


def _write_yaml(dataset_dir: Path) -> None:
    """Genera dataset.yaml dinámicamente con path absoluto."""
    DATASET_YAML.write_text(
        (
            f"path: {dataset_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names: ['licence']\n"
        ),
        encoding="utf-8",
    )
    print(f"[YOLO] dataset.yaml generado en {DATASET_YAML}")


def ensure_dataset_ready() -> tuple[Path, Path]:
    """Garantiza que existe el dataset en formato YOLO con dataset.yaml.

    Devuelve (dataset_dir, dataset_yaml). Si el dataset estaba en VOC, los
    datos convertidos quedan en data/processed/plates_yolo/.
    """
    if _is_yolo_dataset(DATASET_DIR):
        n_train = sum(1 for _ in (DATASET_DIR / "images" / "train").iterdir())
        n_val = sum(1 for _ in (DATASET_DIR / "images" / "val").iterdir())
        n_test = sum(1 for _ in (DATASET_DIR / "images" / "test").iterdir())
        print(f"[YOLO] Dataset YOLO listo: train={n_train} · val={n_val} · test={n_test}")
        if not DATASET_YAML.exists():
            _write_yaml(DATASET_DIR)
        return DATASET_DIR, DATASET_YAML

    if _is_voc_dataset(DATASET_DIR):
        print("[YOLO] Detectado dataset VOC. Convirtiendo a YOLO (70/15/15)...")
        out_dir = ROOT / "data" / "processed" / "plates_yolo"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "data" / "voc_to_yolo.py"),
                "--voc-dir", str(DATASET_DIR),
                "--out-dir", str(out_dir),
                "--classes", "licence",
                "--val-split", "0.15",
                "--test-split", "0.15",
            ],
            check=True,
        )
        return out_dir, out_dir / "dataset.yaml"

    raise FileNotFoundError(
        f"Dataset no encontrado o estructura inválida en {DATASET_DIR}.\n"
        "Esperado: images/{train,val,test}/  +  labels/{train,val,test}/  (YOLO)\n"
        "        ó: annotations/*.xml + images/  (Pascal VOC)"
    )


def train(fast: bool, dataset_dir: Path, dataset_yaml: Path) -> None:
    from ultralytics import YOLO
    import torch

    # YOLOv8s (small): 11.2M params · ~mAP +3-5 pts vs nano.
    # 4 épocas a 100% data (~3h en M4) — con 25K imgs por época el modelo ve
    # mucha variedad. patience=1 corta inmediatamente si no mejora.
    epochs = 6 if fast else 4
    patience = 3 if fast else 1
    fraction = 0.10 if fast else 1.0
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(
        f"[YOLO] FAST_MODE={fast} | model=yolov8s | epochs={epochs} | "
        f"patience={patience} | fraction={fraction} | device={device}"
    )

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="plate_detector",
        patience=patience,
        fraction=fraction,
        device=device,
        exist_ok=True,
        project=str(ROOT / "runs" / "detect"),
    )

    val_metrics = model.val(split="val")
    print(f"[YOLO] [val]  mAP50={val_metrics.box.map50:.3f} | mAP50-95={val_metrics.box.map:.3f}")

    # Evaluación oficial sobre el holdout 'test' (no influye en EarlyStopping).
    test_metrics = None
    test_imgs_dir = dataset_dir / "images" / "test"
    if test_imgs_dir.exists() and any(test_imgs_dir.iterdir()):
        test_metrics = model.val(
            split="test",
            name="plate_detector_test",
            project=str(ROOT / "runs" / "detect"),
        )
        print(f"[YOLO] [test] mAP50={test_metrics.box.map50:.3f} | mAP50-95={test_metrics.box.map:.3f}")
    else:
        print("[YOLO] ⚠ No hay imágenes en split 'test' — métrica oficial será sobre 'val'.")

    best = ROOT / "runs" / "detect" / "plate_detector" / "weights" / "best.pt"
    dst = MODELS_DIR / "plate_detector.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print(f"[YOLO] ✓ Pesos copiados a {dst}")

    # Persistir métricas para /metrics
    summary = {
        "val": {"map50": float(val_metrics.box.map50), "map50_95": float(val_metrics.box.map)},
    }
    if test_metrics is not None:
        summary["test"] = {
            "map50": float(test_metrics.box.map50),
            "map50_95": float(test_metrics.box.map),
        }
    plots_dir = ROOT / "app" / "static" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "yolo_metrics.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Modo demo (~6 épocas, fracción 10%)")
    args = parser.parse_args()

    dataset_dir, dataset_yaml = ensure_dataset_ready()
    train(fast=args.fast, dataset_dir=dataset_dir, dataset_yaml=dataset_yaml)


if __name__ == "__main__":
    main()
