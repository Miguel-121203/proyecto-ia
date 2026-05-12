"""Crea un split estratificado train/val/test para el dataset de vehículos.

Usa **symlinks** (no copia archivos) sobre `data/raw/vehicles/<clase>/*.jpg` y
genera la estructura esperada por `flow_from_directory`:

    data/processed/vehicles_split/
        train/{Bike,Car}/
        val/{Bike,Car}/
        test/{Bike,Car}/

Por defecto: 70% train · 15% val · 15% test (semilla fija = 42).

Uso:
    python scripts/split_vehicles.py
    python scripts/split_vehicles.py --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "raw" / "modelo1"
DST_DIR = ROOT / "data" / "processed" / "vehicles_split"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _split_class(
    class_name: str,
    src_class_dir: Path,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, int]:
    images = sorted(p for p in src_class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No se encontraron imágenes en {src_class_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    counts: dict[str, int] = {}
    for split_name, files in splits.items():
        out_dir = DST_DIR / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in files:
            link = out_dir / img.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(img.resolve())
        counts[split_name] = len(files)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Borra el split previo antes de generar.")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Las proporciones deben sumar 1.0 (suman {total:.3f})")

    if not SRC_DIR.exists():
        raise SystemExit(f"No existe {SRC_DIR}. Verifica que la data esté en data/raw/modelo1/.")

    if args.clean and DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    classes = sorted(d.name for d in SRC_DIR.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No se encontraron subcarpetas de clase en {SRC_DIR}")

    print(f"[SPLIT] Origen: {SRC_DIR}")
    print(f"[SPLIT] Destino: {DST_DIR}")
    print(f"[SPLIT] Ratios: train={args.train} val={args.val} test={args.test} | seed={args.seed}")
    print(f"[SPLIT] Clases detectadas: {classes}")

    grand_total = {"train": 0, "val": 0, "test": 0}
    for cls in classes:
        counts = _split_class(cls, SRC_DIR / cls, (args.train, args.val, args.test), args.seed)
        print(f"[SPLIT]  {cls}: train={counts['train']}  val={counts['val']}  test={counts['test']}")
        for k, v in counts.items():
            grand_total[k] += v

    print(f"[SPLIT] ✓ Total: train={grand_total['train']}  val={grand_total['val']}  test={grand_total['test']}")


if __name__ == "__main__":
    main()
