"""Conversión Pascal VOC (XML) → formato YOLO (txt).

El formato YOLO por imagen:
    <class_id> <x_center> <y_center> <width> <height>
todas las coordenadas normalizadas al rango [0, 1].

Uso:
    python scripts/voc_to_yolo.py \
        --voc-dir data/raw/car-license-plate-detection \
        --out-dir data/processed/plates_yolo \
        --classes license_plate \
        --val-split 0.15

La estructura de salida es la esperada por Ultralytics YOLOv8:
    out-dir/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
        dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _find_voc_dirs(voc_dir: Path) -> tuple[Path, Path]:
    """Localiza las carpetas de imágenes y anotaciones en un árbol VOC.

    Soporta las variantes más comunes:
        - voc_dir/images + voc_dir/annotations
        - voc_dir/JPEGImages + voc_dir/Annotations
        - voc_dir con imágenes y XML mezclados
    """
    candidates_img = ["images", "JPEGImages"]
    candidates_ann = ["annotations", "Annotations"]

    img_dir = next((voc_dir / c for c in candidates_img if (voc_dir / c).exists()), voc_dir)
    ann_dir = next((voc_dir / c for c in candidates_ann if (voc_dir / c).exists()), voc_dir)
    return img_dir, ann_dir


def _parse_voc_xml(xml_path: Path, class_map: dict[str, int]) -> tuple[str, list[str]] | None:
    """Parsea un XML VOC y devuelve (filename, lista de líneas YOLO)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_el = root.find("filename")
    size_el = root.find("size")
    if filename_el is None or size_el is None:
        logger.warning("XML inválido (sin filename/size): %s", xml_path)
        return None

    filename = filename_el.text or ""
    width = int(size_el.findtext("width", "0"))
    height = int(size_el.findtext("height", "0"))
    if width <= 0 or height <= 0:
        logger.warning("Tamaño inválido en %s", xml_path)
        return None

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_map:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w_norm = (xmax - xmin) / width
        h_norm = (ymax - ymin) / height

        cls_id = class_map[name]
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return filename, lines


def _find_image(img_dir: Path, stem: str, declared_filename: str) -> Path | None:
    """Busca la imagen asociada a un XML probando extensiones comunes."""
    declared = img_dir / declared_filename
    if declared.exists():
        return declared
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert(
    voc_dir: Path,
    out_dir: Path,
    classes: list[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    img_dir, ann_dir = _find_voc_dirs(voc_dir)
    logger.info("Imágenes en: %s", img_dir)
    logger.info("Anotaciones en: %s", ann_dir)

    class_map = {name: idx for idx, name in enumerate(classes)}

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No se encontraron XML en {ann_dir}")

    rng = random.Random(seed)
    rng.shuffle(xml_files)

    n_total = len(xml_files)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    val_set = set(xml_files[:n_val])
    test_set = set(xml_files[n_val : n_val + n_test])
    logger.info(
        "Total XML: %d | train: %d | val: %d | test: %d",
        n_total, n_total - n_val - n_test, n_val, n_test,
    )

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    for xml_path in xml_files:
        parsed = _parse_voc_xml(xml_path, class_map)
        if parsed is None:
            stats["skipped"] += 1
            continue
        filename, lines = parsed
        if not lines:
            stats["skipped"] += 1
            continue

        img_src = _find_image(img_dir, xml_path.stem, filename)
        if img_src is None:
            logger.warning("Imagen no encontrada para %s", xml_path.name)
            stats["skipped"] += 1
            continue

        if xml_path in val_set:
            split = "val"
        elif xml_path in test_set:
            split = "test"
        else:
            split = "train"
        stats[split] += 1

        img_dst = out_dir / "images" / split / img_src.name
        lbl_dst = out_dir / "labels" / split / f"{img_src.stem}.txt"

        shutil.copy2(img_src, img_dst)
        lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        (
            f"path: {out_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"nc: {len(classes)}\n"
            f"names: {classes}\n"
        ),
        encoding="utf-8",
    )

    logger.info("Conversión finalizada: %s", stats)
    logger.info("dataset.yaml: %s", yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte anotaciones Pascal VOC a formato YOLO.")
    parser.add_argument("--voc-dir", type=Path, required=True, help="Carpeta raíz VOC.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Salida estilo Ultralytics.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["license_plate"],
        help="Lista ordenada de clases (el índice define class_id).",
    )
    parser.add_argument("--val-split", type=float, default=0.15, help="Fracción para validación.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fracción para test (holdout).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_split + args.test_split >= 1.0:
        raise SystemExit("val-split + test-split debe ser < 1.0")

    convert(
        voc_dir=args.voc_dir,
        out_dir=args.out_dir,
        classes=args.classes,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
