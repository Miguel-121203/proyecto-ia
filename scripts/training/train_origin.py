"""Entrenamiento del Modelo 3 — clasificador de estado USA (MobileNetV2).

Dataset: data/raw/modelo3/us_license_plates/
  Estructura: train/ valid/ test/  (una subcarpeta por clase en cada split)
  51 clases · ~25 723 imgs train · 505 valid · 505 test

Arquitectura: MobileNetV2 con input personalizado 300×300 (mejor que el default
              224 para placas). 3.5M params, converge bien en pocas épocas.
              Reanuda automáticamente si el proceso se interrumpe.

Uso desde la raíz del proyecto:
    python scripts/train_origin.py                  # completo (reanuda si hay checkpoint)
    python scripts/train_origin.py --fast           # modo demo
    python scripts/train_origin.py --reset          # borra estado y empieza desde cero
    python scripts/train_origin.py --resume-finetune  # continúa ft con más capas y lr menor
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "modelo3" / "us_license_plates"
PLOTS_DIR = ROOT / "app" / "static" / "plots"
MODELS_DIR = ROOT / "models"

IMG_SIZE = 300  # input personalizado para MobileNetV2 (mejor que 224 default en placas)
SEED = 42

# Archivos de reanudación (se borran al terminar con éxito)
STATE_PATH = MODELS_DIR / "origin_train_state.json"
CHECKPOINT_PATH = MODELS_DIR / "origin_checkpoint.keras"


# ── Estado de reanudación ─────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "phase": "head",
        "phase_epoch": 0,
        "best_head_acc": 0.0,
        "history_head": {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []},
        "history_ft":   {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []},
    }


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


class _ResumeCallback(tf.keras.callbacks.Callback):
    """Persiste el estado del entrenamiento al final de cada época."""

    def __init__(self, state: dict, phase: str, hist_key: str) -> None:
        super().__init__()
        self.state = state
        self.phase = phase
        self.hist_key = hist_key

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        h = self.state[self.hist_key]
        h["accuracy"].append(float(logs.get("accuracy", 0)))
        h["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
        h["loss"].append(float(logs.get("loss", 0)))
        h["val_loss"].append(float(logs.get("val_loss", 0)))
        self.state["phase"] = self.phase
        self.state["phase_epoch"] = len(h["accuracy"])
        self.model.save(str(CHECKPOINT_PATH))
        _save_state(self.state)


# ── Construcción del modelo ───────────────────────────────────────────────────

def _build_model(num_classes: int) -> tuple[Model, tf.keras.Model]:
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(base.input, out), base


def _unfreeze_top(model: Model, n: int = 20) -> None:
    """Descongela las últimas n capas del backbone (todo lo anterior al GlobalAveragePooling2D)."""
    head_start = next(
        i for i, l in enumerate(model.layers)
        if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
    )
    backbone_layers = model.layers[:head_start]
    for layer in backbone_layers[:-n]:
        layer.trainable = False
    for layer in backbone_layers[-n:]:
        layer.trainable = True


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train(fast: bool, reset: bool, resume_finetune: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        for p in (STATE_PATH,):
            if p.exists():
                p.unlink()
        if CHECKPOINT_PATH.exists():
            shutil.rmtree(str(CHECKPOINT_PATH), ignore_errors=True)
            CHECKPOINT_PATH.unlink(missing_ok=True)
        print("[ORIGIN] Estado anterior eliminado. Empezando desde cero.")

    if fast:
        batch_size, epochs_head, epochs_ft = 32, 2, 3
        steps_per_epoch, val_steps = 20, 5
    else:
        # MobileNetV2 300×300 — config que produjo el modelo de referencia con
        # Top-1 87.7% en test: ~15 head + 50 ft con 20 capas = 65 épocas base.
        # Luego ejecutar --resume-finetune para 15 épocas más con 40 capas → ~80.
        # EarlyStopping (patience=6 head / 8 ft) corta antes si converge.
        batch_size, epochs_head, epochs_ft = 32, 15, 50
        steps_per_epoch, val_steps = None, None

    if resume_finetune:
        epochs_ft = max(epochs_ft, _load_state().get("phase_epoch", 0) + 15)

    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"
    test_dir = DATA_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {train_dir}.\n"
            "Ejecuta primero: python scripts/combine_plates.py"
        )
    has_test_split = test_dir.exists() and any(test_dir.iterdir())

    state = _load_state()
    resuming = state["phase_epoch"] > 0 or state["phase"] == "ft"
    if resuming:
        print(f"[ORIGIN] ▶ Reanudando — fase '{state['phase']}', época {state['phase_epoch']}")
    else:
        print(f"[ORIGIN] FAST={fast} | batch={batch_size} | head={epochs_head} épocas + ft={epochs_ft} épocas")

    print(f"[ORIGIN] GPU disponibles: {tf.config.list_physical_devices('GPU')}")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        brightness_range=(0.75, 1.25),
        channel_shift_range=15.0,
        horizontal_flip=False,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    class_names = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]
    print(f"[ORIGIN] {num_classes} clases detectadas")

    with open(MODELS_DIR / "origin_classes.json", "w") as f:
        json.dump(class_names, f, indent=2)

    best_path = str(MODELS_DIR / "origin_classifier_best.h5")

    # ── Fase 1: Head (backbone congelado) ─────────────────────────────────────
    head_complete = state["phase"] == "ft" or state["phase_epoch"] >= epochs_head

    if not head_complete:
        initial_epoch = state["phase_epoch"]

        if CHECKPOINT_PATH.exists() and initial_epoch > 0:
            print(f"[ORIGIN] Cargando checkpoint de fase head (época {initial_epoch})")
            model = tf.keras.models.load_model(str(CHECKPOINT_PATH), compile=False)
        else:
            model, _ = _build_model(num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )
        print(f"[ORIGIN] 1/2 · Head | épocas {initial_epoch} → {epochs_head}")

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_head,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
                _ResumeCallback(state, "head", "history_head"),
            ],
        )

        best_head_acc = max(state["history_head"]["val_accuracy"])
        state["best_head_acc"] = best_head_acc
        state["phase"] = "ft"
        state["phase_epoch"] = 0
        _save_state(state)

    else:
        best_head_acc = state["best_head_acc"]
        print(f"[ORIGIN] 1/2 · Head ya completada | best_val_acc={best_head_acc:.4f}")

    # ── Fase 2: Fine-tuning ───────────────────────────────────────────────────
    ft_initial = state["phase_epoch"] if state["phase"] == "ft" else 0

    if ft_initial >= epochs_ft and not resume_finetune:
        print(f"[ORIGIN] Fine-tuning ya completado ({ft_initial} épocas).")
        model = tf.keras.models.load_model(best_path, compile=False)
    else:
        if resume_finetune:
            # Carga el mejor checkpoint (no el último) y descongela más capas con lr menor
            print(f"[ORIGIN] ▶ --resume-finetune: cargando mejor checkpoint | 40 capas | lr=1e-5")
            model = tf.keras.models.load_model(best_path, compile=False)
            _unfreeze_top(model, n=40)
            ft_lr = 1e-5
        elif CHECKPOINT_PATH.exists() and ft_initial > 0:
            print(f"[ORIGIN] Cargando checkpoint de fase ft (época {ft_initial})")
            model = tf.keras.models.load_model(str(CHECKPOINT_PATH), compile=False)
            ft_lr = 3e-5
        else:
            # Transición head → ft: cargar mejor head y descongelar
            if CHECKPOINT_PATH.exists():
                model = tf.keras.models.load_model(str(CHECKPOINT_PATH), compile=False)
            else:
                model, _ = _build_model(num_classes)
                model.load_weights(best_path)
            _unfreeze_top(model, n=20)
            ft_lr = 3e-5

        model.compile(
            optimizer=tf.keras.optimizers.Adam(ft_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )
        print(f"[ORIGIN] 2/2 · Fine-tuning | épocas {ft_initial} → {epochs_ft} | lr={ft_lr} | best_head={best_head_acc:.4f}")

        # Umbral: no sobreescribir el mejor checkpoint si ya tiene buena val_acc
        current_best = max(state["history_ft"]["val_accuracy"]) if state["history_ft"]["val_accuracy"] else best_head_acc

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_ft,
            initial_epoch=ft_initial,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8),
                ModelCheckpoint(
                    best_path,
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1,
                    initial_value_threshold=current_best,
                ),
                _ResumeCallback(state, "ft", "history_ft"),
            ],
        )

    # ── Evaluación final sobre el mejor modelo guardado ───────────────────────
    best_model = tf.keras.models.load_model(best_path, compile=False)

    # Evaluación sobre val (sanity check) y, si existe, sobre test (métrica oficial).
    eval_splits: list[tuple[str, Path]] = [("val", valid_dir)]
    if has_test_split:
        eval_splits.append(("test", test_dir))
    else:
        print("[ORIGIN] ⚠ Sin split 'test' en el dataset — la métrica oficial será sobre 'val'.")

    final_metrics: dict[str, dict[str, float]] = {}

    for split_name, split_dir in eval_splits:
        eval_gen = val_datagen.flow_from_directory(
            split_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
            classes=class_names,
        )
        probs = best_model.predict(eval_gen, verbose=1)
        y_pred = probs.argmax(axis=1)
        y_true = eval_gen.classes[: len(y_pred)]

        print(f"\n[ORIGIN] === Reporte sobre split '{split_name}' ===")
        print(
            classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                target_names=class_names,
                zero_division=0,
            )
        )

        top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        top5.update_state(tf.one_hot(y_true, num_classes), probs[: len(y_true)])
        top5_acc = float(top5.result())
        top1_acc = float((y_pred == y_true).mean())
        final_metrics[split_name] = {"top1": top1_acc, "top5": top5_acc}
        print(f"[ORIGIN] [{split_name}] Top-1: {top1_acc:.4f} | Top-5: {top5_acc:.4f}")

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
            + (" (FAST)" if fast else "")
        )
        plt.tight_layout()
        suffix = "" if split_name == "val" else f"_{split_name}"
        plt.savefig(PLOTS_DIR / f"origin_confusion_matrix{suffix}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Métricas oficiales: prefiere 'test' si está disponible.
    official_split = "test" if "test" in final_metrics else "val"
    top1_val = final_metrics[official_split]["top1"]
    top5_val = final_metrics[official_split]["top5"]
    print(f"\n[ORIGIN] ► Métricas oficiales ({official_split}): "
          f"Top-1 {top1_val:.4f} | Top-5 {top5_val:.4f}")

    # History JSON para Chart.js
    acc      = state["history_head"]["accuracy"]     + state["history_ft"]["accuracy"]
    val_acc  = state["history_head"]["val_accuracy"] + state["history_ft"]["val_accuracy"]
    loss     = state["history_head"]["loss"]          + state["history_ft"]["loss"]
    val_loss = state["history_head"]["val_loss"]      + state["history_ft"]["val_loss"]

    with open(PLOTS_DIR / "history_origin.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "val_accuracy": val_acc,
            "loss": loss,
            "val_loss": val_loss,
            "top5_val_accuracy": top5_val,
            "num_classes": num_classes,
            "official_split": official_split,
            "final_metrics": final_metrics,
        }, f)

    dst = MODELS_DIR / "origin_classifier.h5"
    best_model.save(dst)
    print(f"[ORIGIN] ✓ Modelo final guardado en {dst}")
    print(f"[ORIGIN] ✓ Mejor checkpoint en {best_path}")

    # Limpiar archivos de reanudación al terminar con éxito
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    for p in (CHECKPOINT_PATH,):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(str(p))
            else:
                p.unlink()
    print("[ORIGIN] ✓ Archivos de reanudación eliminados.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Modo demo rápido.")
    parser.add_argument("--reset", action="store_true", help="Borra el estado de reanudación y empieza desde cero.")
    parser.add_argument("--resume-finetune", action="store_true", help="Continúa ft desde el mejor checkpoint con 40 capas y lr=1e-5.")
    args = parser.parse_args()
    train(fast=args.fast, reset=args.reset, resume_finetune=args.resume_finetune)


if __name__ == "__main__":
    main()
