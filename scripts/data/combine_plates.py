"""Combina plates y new plates en un único dataset con symlinks.

Fuentes:
  A) /Users/miguel/Downloads/archive/plates/       → 50 clases (estados)
  B) /Users/miguel/Downloads/archive/new plates/   → 56 clases (estados + territorios)

Resultado en data/raw/us_license_plates_combined/:
  - 51 clases en train/valid/test (50 estados + WASHINGTON DC)
  - Territorios excluidos por decisión de diseño (ver EXCLUDED_CLASSES)
  - Para las 50 clases compartidas: imágenes de A + B (sin duplicar disco)

Uso:
    python scripts/combine_plates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELO3 = ROOT / "data" / "raw" / "modelo3"
FUENTES = MODELO3 / "fuentes"
SRC_A = FUENTES / "gpiosenka_50_estados"
SRC_B = FUENTES / "gpiosenka_56_clases_territorios"
DST = MODELO3 / "us_license_plates"

SPLITS = ("train", "valid", "test")

# Territorios excluidos: el sistema apunta a placas de estados USA + DC.
# Las placas de estos territorios son extremadamente raras en parqueaderos
# corporativos del continente, y CNMI tiene muy pocas imágenes (~86).
EXCLUDED_CLASSES: set[str] = {
    "AMERICAN SAMOA",
    "CNMI",
    "GUAM",
    "PUERTO RICO",
    "U S VIRGIN ISLANDS",
}


def link_images(src_dir: Path, dst_dir: Path, prefix: str) -> int:
    """Crea symlinks de todas las imágenes de src_dir en dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        link = dst_dir / f"{prefix}_{img.name}"
        if not link.exists():
            link.symlink_to(img.resolve())
        count += 1
    return count


def combine() -> None:
    print(f"[COMBINE] Destino: {DST}")

    for split in SPLITS:
        dir_a = SRC_A / split
        dir_b = SRC_B / split

        if not dir_b.exists():
            print(f"[WARN] {dir_b} no existe, saltando split '{split}'.")
            continue

        # Recopilar todas las clases (B es el superset con 56), filtrando territorios
        classes_b = {d.name for d in dir_b.iterdir() if d.is_dir()} - EXCLUDED_CLASSES
        classes_a = (
            {d.name for d in dir_a.iterdir() if d.is_dir()} - EXCLUDED_CLASSES
            if dir_a.exists() else set()
        )

        total_imgs = 0
        for cls in sorted(classes_b):
            dst_cls = DST / split / cls

            # Imágenes de B
            n_b = link_images(dir_b / cls, dst_cls, prefix="B")

            # Imágenes de A (solo si la clase existe en ambos)
            n_a = 0
            if cls in classes_a:
                n_a = link_images(dir_a / cls, dst_cls, prefix="A")

            total_imgs += n_a + n_b

        only_a = classes_a - classes_b
        if only_a:
            print(f"[WARN] Clases en A pero no en B para split '{split}': {only_a}")

        print(f"[COMBINE] {split}: {len(classes_b)} clases · {total_imgs} imágenes")

    print("[COMBINE] ✓ Listo.")


def main() -> None:
    for src in (SRC_A, SRC_B):
        if not src.exists():
            print(f"[ERROR] No encontrado: {src}")
            sys.exit(1)

    combine()

    # Mostrar resumen
    for split in SPLITS:
        split_dir = DST / split
        if split_dir.exists():
            n_cls = sum(1 for d in split_dir.iterdir() if d.is_dir())
            n_img = sum(1 for d in split_dir.iterdir() if d.is_dir()
                        for _ in d.iterdir())
            print(f"  {split}: {n_cls} clases, {n_img} imágenes")


if __name__ == "__main__":
    main()
