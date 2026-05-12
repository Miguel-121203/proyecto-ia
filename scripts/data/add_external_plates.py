"""Integra datasets externos al train set de us_license_plates.

Fuentes:
  C) tolgadincer  ~/Downloads/archive-3/data/          → 51 clases, 4 463 imgs
  D) jonkeegan    ~/Downloads/us-license-plates/plates/ → 51 clases, 8 239 imgs
     (jonkeegan incluye GIFs animados que se descartan automáticamente)

Solo se tocan las carpetas train/. Valid y test quedan intactos.
Las imágenes se agregan via symlinks (sin duplicar disco).

Uso:
    python scripts/add_external_plates.py
    python scripts/add_external_plates.py --dry-run   # solo cuenta, no crea links
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELO3 = ROOT / "data" / "raw" / "modelo3"
DATA_DIR = MODELO3 / "us_license_plates" / "train"
FUENTES = MODELO3 / "fuentes"
SRC_TOLGADINCER = FUENTES / "tolgadincer_50_estados"
SRC_JONKEEGAN   = FUENTES / "jonkeegan_51_estados_foia" / "plates"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}   # GIF excluido intencionalmente

# ── Mapeo tolgadincer → nombre de clase en nuestro dataset ───────────────────
TOLGADINCER_MAP: dict[str, str] = {
    "Alabama": "ALABAMA", "Alaska": "ALASKA", "Arizona": "ARIZONA",
    "Arkansas": "ARKANSAS", "California": "CALIFORNIA", "Colorado": "COLORADO",
    "Connecticut": "CONNECTICUT", "Delaware": "DELAWARE", "Florida": "FLORIDA",
    "Georgia": "GEORGIA",
    "Hawaii": "HAWAI",          # typo del dataset original — mantener consistencia
    "Idaho": "IDAHO", "Illinois": "ILLINOIS", "Indiana": "INDIANA",
    "Iowa": "IOWA", "Kansas": "KANSAS", "Kentucky": "KENTUCKY",
    "Louisiana": "LOUISIANA", "Maine": "MAINE", "Maryland": "MARYLAND",
    "Massachusetts": "MASSACHUSETTS", "Michigan": "MICHIGAN",
    "Minnesota": "MINNESOTA",
    "Mississippi": "MISSIPPI",  # typo del dataset original — mantener consistencia
    "Missouri": "MISSOURI", "Montana": "MONTANA", "Nebraska": "NEBRASKA",
    "Nevada": "NEVADA", "NewHampshire": "NEW HAMPSHIRE",
    "NewJersey": "NEW JERSEY", "NewMexico": "NEW MEXICO",
    "NewYork": "NEW YORK", "NorthCarolina": "NORTH CAROLINA",
    "NorthDakota": "NORTH DAKOTA", "Ohio": "OHIO", "Oklahoma": "OKLAHOMA",
    "Oregon": "OREGON", "Pennsylvania": "PENNSYLVANIA",
    "RhodeIsland": "RHODE ISLAND", "SouthCarolina": "SOUTH CAROLINA",
    "SouthDakota": "SOUTH DAKOTA", "Tennessee": "TENNESSEE",
    "Texas": "TEXAS", "Utah": "UTAH", "Vermont": "VERMONT",
    "Virginia": "VIRGINIA", "Washington": "WASHINGTON",
    "WashingtonDC": "WASHINGTON DC", "WestVirginia": "WEST VIRGINIA",
    "Wisconsin": "WISCONSIN", "Wyoming": "WYOMING",
}

# ── Mapeo jonkeegan (abreviaciones) → nombre de clase ────────────────────────
JONKEEGAN_MAP: dict[str, str] = {
    "AK": "ALASKA",    "AL": "ALABAMA",      "AR": "ARKANSAS",
    "AZ": "ARIZONA",   "CA": "CALIFORNIA",   "CO": "COLORADO",
    "CT": "CONNECTICUT","DC": "WASHINGTON DC","DE": "DELAWARE",
    "FL": "FLORIDA",   "GA": "GEORGIA",
    "HI": "HAWAI",     # typo del dataset original
    "IA": "IOWA",      "ID": "IDAHO",        "IL": "ILLINOIS",
    "IN": "INDIANA",   "KS": "KANSAS",       "KY": "KENTUCKY",
    "LA": "LOUISIANA", "MA": "MASSACHUSETTS","MD": "MARYLAND",
    "ME": "MAINE",     "MI": "MICHIGAN",     "MN": "MINNESOTA",
    "MO": "MISSOURI",
    "MS": "MISSIPPI",  # typo del dataset original
    "MT": "MONTANA",   "NC": "NORTH CAROLINA","ND": "NORTH DAKOTA",
    "NE": "NEBRASKA",  "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY",
    "NM": "NEW MEXICO","NV": "NEVADA",        "NY": "NEW YORK",
    "OH": "OHIO",      "OK": "OKLAHOMA",      "OR": "OREGON",
    "PA": "PENNSYLVANIA","RI": "RHODE ISLAND","SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA","TN": "TENNESSEE",   "TX": "TEXAS",
    "UT": "UTAH",      "VA": "VIRGINIA",      "VT": "VERMONT",
    "WA": "WASHINGTON","WI": "WISCONSIN",     "WV": "WEST VIRGINIA",
    "WY": "WYOMING",
}


def link_images(src_dir: Path, dst_dir: Path, prefix: str, dry_run: bool) -> int:
    """Crea symlinks de las imágenes de src_dir en dst_dir."""
    added = 0
    for img in src_dir.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        link = dst_dir / f"{prefix}_{img.name}"
        if link.exists():
            continue
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            link.symlink_to(img.resolve())
        added += 1
    return added


def integrate(src_root: Path, name_map: dict[str, str], prefix: str, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src_cls_dir in sorted(src_root.iterdir()):
        if not src_cls_dir.is_dir():
            continue
        our_cls = name_map.get(src_cls_dir.name)
        if our_cls is None:
            print(f"  [WARN] Sin mapeo para '{src_cls_dir.name}' — saltando")
            continue
        dst = DATA_DIR / our_cls
        n = link_images(src_cls_dir, dst, prefix, dry_run)
        counts[our_cls] = n
    return counts


def main(dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "REAL"
    print(f"[ADD] Modo: {mode}")

    for src in (SRC_TOLGADINCER, SRC_JONKEEGAN):
        if not src.exists():
            print(f"[ERROR] No encontrado: {src}")
            sys.exit(1)

    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset train no encontrado: {DATA_DIR}")
        sys.exit(1)

    # ── tolgadincer ───────────────────────────────────────────────────────────
    print(f"\n[ADD] Integrando tolgadincer ({SRC_TOLGADINCER}) …")
    counts_t = integrate(SRC_TOLGADINCER, TOLGADINCER_MAP, "T", dry_run)
    total_t = sum(counts_t.values())
    for cls, n in sorted(counts_t.items()):
        if n > 0:
            print(f"  {cls}: +{n}")
    print(f"  → {total_t} imágenes nuevas de tolgadincer")

    # ── jonkeegan ─────────────────────────────────────────────────────────────
    print(f"\n[ADD] Integrando jonkeegan ({SRC_JONKEEGAN}) …")
    counts_j = integrate(SRC_JONKEEGAN, JONKEEGAN_MAP, "J", dry_run)
    total_j = sum(counts_j.values())
    for cls, n in sorted(counts_j.items()):
        if n > 0:
            print(f"  {cls}: +{n}")
    print(f"  → {total_j} imágenes nuevas de jonkeegan")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n[ADD] ✓ Total nuevas imágenes: {total_t + total_j}")
    print("[ADD] ✓ Valid y test NO modificados.")
    if dry_run:
        print("[ADD] (dry-run: no se crearon symlinks)")
    else:
        # Conteo final del train set
        total_train = sum(
            1 for cls_dir in DATA_DIR.iterdir() if cls_dir.is_dir()
            for _ in cls_dir.iterdir()
        )
        print(f"[ADD] ✓ Train set total ahora: {total_train} imágenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta, no crea symlinks.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
