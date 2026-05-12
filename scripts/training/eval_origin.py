"""Evalúa un origin_classifier.h5 ya entrenado sobre val y test.

Uso desde la raíz del proyecto:
    python scripts/training/eval_origin.py
    python scripts/training/eval_origin.py --weights models/origin_classifier_best.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mbnv2_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "modelo3" / "us_license_plates"
PLOTS_DIR = ROOT / "app" / "static" / "plots"
MODELS_DIR = ROOT / "models"

BATCH_SIZE = 32


def _detect_backbone(model: tf.keras.Model) -> tuple[int, callable]:
    """Detecta IMG_SIZE y preprocess function según el backbone del modelo."""
    img_size = model.input_shape[1]
    layer_names = {l.name for l in model.layers[:20]}
    is_mobilenet = "Conv1" in layer_names or "expanded_conv_depthwise" in layer_names
    preprocess = mbnv2_preprocess if is_mobilenet else effv2_preprocess
    backbone = "MobileNetV2" if is_mobilenet else "EfficientNetV2B0"
    print(f"[EVAL] Backbone detectado: {backbone} | input {img_size}×{img_size}")
    return img_size, preprocess


def evaluate(weights_path: Path) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {weights_path}")

    print(f"[EVAL] Cargando modelo desde {weights_path}")
    model = tf.keras.models.load_model(str(weights_path), compile=False)

    img_size, preprocess_input = _detect_backbone(model)

    classes_path = MODELS_DIR / "origin_classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"No se encontró {classes_path}")
    class_names = json.loads(classes_path.read_text())
    num_classes = len(class_names)
    print(f"[EVAL] {num_classes} clases cargadas")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    splits = [("val", DATA_DIR / "valid"), ("test", DATA_DIR / "test")]
    final_metrics: dict[str, dict[str, float]] = {}

    for split_name, split_dir in splits:
        if not split_dir.exists():
            print(f"[EVAL] ⚠ Saltando '{split_name}' — no existe {split_dir}")
            continue

        gen = val_datagen.flow_from_directory(
            split_dir,
            target_size=(img_size, img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
            classes=class_names,
        )
        probs = model.predict(gen, verbose=1)
        y_pred = probs.argmax(axis=1)
        y_true = gen.classes[: len(y_pred)]

        top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        top5.update_state(tf.one_hot(y_true, num_classes), probs[: len(y_true)])
        top5_acc = float(top5.result())
        top1_acc = float((y_pred == y_true).mean())
        final_metrics[split_name] = {"top1": top1_acc, "top5": top5_acc}

        print(f"\n[EVAL] === Reporte sobre split '{split_name}' ===")
        print(
            classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                target_names=class_names,
                zero_division=0,
            )
        )
        print(f"[EVAL] [{split_name}] Top-1: {top1_acc:.4f} | Top-5: {top5_acc:.4f}")

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.title(
            f"Matriz de confusión [{split_name}] — clasificador de estado USA ({num_classes} clases)"
            f" · Top-1 {top1_acc*100:.1f}% | Top-5 {top5_acc*100:.1f}%"
        )
        plt.tight_layout()
        suffix = "" if split_name == "val" else f"_{split_name}"
        plt.savefig(PLOTS_DIR / f"origin_confusion_matrix{suffix}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    if "test" in final_metrics:
        official = final_metrics["test"]
        split_name = "test"
    elif "val" in final_metrics:
        official = final_metrics["val"]
        split_name = "val"
    else:
        print("[EVAL] No se pudo evaluar ningún split.")
        return

    print(f"\n[EVAL] ► Métricas oficiales ({split_name}): "
          f"Top-1 {official['top1']:.4f} | Top-5 {official['top5']:.4f}")

    # history_origin.json mínimo para que /metrics muestre los KPIs y matrices
    # (sin curvas por época: el modelo se entrenó fuera de esta sesión)
    history_path = PLOTS_DIR / "history_origin.json"
    existing = json.loads(history_path.read_text()) if history_path.exists() else {}
    history = {
        "accuracy": existing.get("accuracy", []),
        "val_accuracy": existing.get("val_accuracy", []),
        "loss": existing.get("loss", []),
        "val_loss": existing.get("val_loss", []),
        "num_classes": num_classes,
        "official_split": split_name,
        "final_metrics": final_metrics,
        "top5_val_accuracy": final_metrics.get("val", {}).get("top5"),
    }
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[EVAL] ✓ Métricas guardadas en {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=MODELS_DIR / "origin_classifier_best.h5",
        help="Ruta al .h5 a evaluar (default: models/origin_classifier_best.h5)",
    )
    args = parser.parse_args()
    evaluate(args.weights)


if __name__ == "__main__":
    main()
