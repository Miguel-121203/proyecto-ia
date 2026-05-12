"""Entrenamiento del Modelo 1 (MobileNetV2) con progreso en vivo.

Uso desde la raíz del proyecto:
    python scripts/train_vehicle.py            # modo completo
    python scripts/train_vehicle.py --fast     # modo demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin GUI; solo guardamos PNG

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "vehicles_split"
LEGACY_DATA_DIR = ROOT / "data" / "raw" / "vehicles"
PLOTS_DIR = ROOT / "app" / "static" / "plots"
MODELS_DIR = ROOT / "models"

IMG_SIZE = 224
SEED = 42


def train(fast: bool) -> None:
    if fast:
        batch_size, epochs_head, epochs_ft = 16, 2, 2
        steps_per_epoch, val_steps = 6, 2
    else:
        # MobileNetV2 binario llega a 99 %+ con muy pocas épocas; bajo a 6+3.
        batch_size, epochs_head, epochs_ft = 32, 6, 3
        steps_per_epoch, val_steps = None, None

    print(f"[VEHICLE] FAST={fast} | batch={batch_size} | epochs={epochs_head} head + {epochs_ft} ft")
    print(f"[VEHICLE] GPU: {tf.config.list_physical_devices('GPU')}")

    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "val"
    test_dir = SPLIT_DIR / "test"

    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise FileNotFoundError(
            f"Splits no encontrados en {SPLIT_DIR}.\n"
            "Ejecuta primero: python scripts/split_vehicles.py"
        )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    print(f"[VEHICLE] Clases: {train_gen.class_indices}")
    print(f"[VEHICLE] Imgs · train={train_gen.samples} val={val_gen.samples} test={test_gen.samples}")

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(2, activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("[VEHICLE] 1/2 · Head (backbone congelado)")
    hist_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_head,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    print("[VEHICLE] 2/2 · Fine-tuning (últimas 20 capas)")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_ft,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Bike (moto)", "Car (carro)"]
    final_metrics: dict[str, float] = {}

    for split_name, gen in (("val", val_gen), ("test", test_gen)):
        gen.reset()
        # Evaluar SIEMPRE el split completo (no aplicar val_steps aquí), si no
        # las métricas finales quedarían sesgadas hacia la primera clase.
        y_pred = np.argmax(model.predict(gen, verbose=1), axis=1)
        y_true = gen.classes[: len(y_pred)]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        acc = float((y_pred == y_true).mean())
        final_metrics[split_name] = acc
        plt.title(f"Matriz de confusión [{split_name}] — vehículo · acc {acc*100:.2f}%")
        plt.tight_layout()
        suffix = "" if split_name == "val" else f"_{split_name}"
        plt.savefig(PLOTS_DIR / f"vehicle_confusion_matrix{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[VEHICLE] === Reporte sobre split '{split_name}' (acc {acc*100:.2f}%) ===")
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=labels, zero_division=0))

    print(f"[VEHICLE] ► val_acc={final_metrics['val']*100:.2f}% | test_acc={final_metrics['test']*100:.2f}%")

    acc = hist_head.history["accuracy"] + hist_ft.history["accuracy"]
    val_acc = hist_head.history["val_accuracy"] + hist_ft.history["val_accuracy"]
    loss = hist_head.history["loss"] + hist_ft.history["loss"]
    val_loss = hist_head.history["val_loss"] + hist_ft.history["val_loss"]

    with open(PLOTS_DIR / "history_vehicle.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "val_accuracy": val_acc,
            "loss": loss,
            "val_loss": val_loss,
            "final_metrics": final_metrics,
        }, f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].plot(acc, color="#0d6efd", linewidth=2, label="train")
    axes[0].plot(val_acc, color="#198754", linewidth=2, label="val")
    axes[0].axvline(epochs_head - 0.5, color="#6c757d", linestyle="--", label="inicio fine-tune")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("época")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    axes[1].plot(loss, color="#dc3545", linewidth=2, label="train")
    axes[1].plot(val_loss, color="#fd7e14", linewidth=2, label="val")
    axes[1].axvline(epochs_head - 0.5, color="#6c757d", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("época")
    axes[1].legend(frameon=False)
    plt.suptitle("Modelo 1 · curvas de entrenamiento" + (" (FAST)" if fast else ""), fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves_vehicle.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / "vehicle_classifier.h5"
    model.save(dst)
    print(f"[VEHICLE] ✓ Guardado en {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Modo demo rápido.")
    args = parser.parse_args()
    train(fast=args.fast)


if __name__ == "__main__":
    main()
