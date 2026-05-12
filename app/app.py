"""Flask app — servicio web del pipeline.

Rutas:
    GET  /         → formulario HTML para subir imagen.
    POST /predict  → recibe multipart/form-data, devuelve JSON.
    GET  /metrics  → renderiza las 3 gráficas obligatorias.

Los modelos se cargan una única vez al arrancar el proceso.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

# Asegurar que src/ sea importable al ejecutar con `python app/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import VehiclePipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(
    __name__,
    template_folder=str(ROOT / "app" / "templates"),
    static_folder=str(ROOT / "app" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

PIPELINE: VehiclePipeline | None = None


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _init_pipeline() -> VehiclePipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = VehiclePipeline()
    return PIPELINE


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Campo 'image' requerido"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Archivo vacío"}), 400

    filename = secure_filename(file.filename)
    if not _allowed(filename):
        return jsonify({"error": f"Formato no soportado. Use {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"No se pudo leer la imagen: {exc}"}), 400

    image_array = np.array(img, dtype=np.uint8)

    try:
        pipeline = _init_pipeline()
        result = pipeline.predict(image_array)
    except NotImplementedError as exc:
        logger.warning("Modelo no disponible: %s", exc)
        return jsonify({"error": "Modelos aún no entrenados", "detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo durante la inferencia")
        return jsonify({"error": "Error interno", "detail": str(exc)}), 500

    return jsonify(result.to_dict())


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _load_yolo_history(path: Path) -> dict | None:
    """Lee runs/detect/.../results.csv y extrae las series por época."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        epochs = [int(float(r["epoch"])) for r in rows]
        precision = [float(r["metrics/precision(B)"]) for r in rows]
        recall = [float(r["metrics/recall(B)"]) for r in rows]
        map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
        map5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        box_loss = [float(r["train/box_loss"]) for r in rows]
        val_box_loss = [float(r["val/box_loss"]) for r in rows]
        return {
            "epochs": epochs,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map5095,
            "box_loss": box_loss,
            "val_box_loss": val_box_loss,
            "final": {
                "precision": precision[-1],
                "recall": recall[-1],
                "map50": map50[-1],
                "map50_95": map5095[-1],
            },
        }
    except Exception:  # noqa: BLE001
        return None


@app.route("/metrics", methods=["GET"])
def metrics():
    plots_dir = ROOT / "app" / "static" / "plots"
    vehicle_hist = _load_json(plots_dir / "history_vehicle.json")
    origin_hist = _load_json(plots_dir / "history_origin.json")
    yolo_hist = _load_yolo_history(plots_dir / "yolo_results.csv")
    yolo_final = _load_json(plots_dir / "yolo_metrics.json")  # val + test post-train

    def _best(values: list[float] | None) -> float | None:
        return max(values) if values else None

    # Métricas oficiales: prefiere split test si existe; si no, val.
    def _origin_official(hist: dict | None) -> tuple[float | None, float | None]:
        if not hist:
            return None, None
        fm = hist.get("final_metrics") or {}
        chosen = fm.get("test") or fm.get("val")
        if chosen:
            return chosen.get("top1"), chosen.get("top5")
        return _best(hist.get("val_accuracy")), hist.get("top5_val_accuracy")

    def _yolo_official() -> tuple[float | None, float | None]:
        if yolo_final and "test" in yolo_final:
            return yolo_final["test"]["map50"], yolo_final["test"]["map50_95"]
        if yolo_hist:
            return yolo_hist["final"]["map50"], yolo_hist["final"]["map50_95"]
        return None, None

    yolo_map50, yolo_map5095 = _yolo_official()

    vehicle_fm = vehicle_hist.get("final_metrics") if vehicle_hist else None
    vehicle_test_acc = vehicle_fm.get("test") if vehicle_fm else None
    vehicle_val_acc = vehicle_fm.get("val") if vehicle_fm else _best(
        vehicle_hist.get("val_accuracy") if vehicle_hist else None
    )
    if vehicle_test_acc is None:
        vehicle_test_acc = vehicle_val_acc

    origin_fm = (origin_hist.get("final_metrics") or {}) if origin_hist else {}
    origin_test = origin_fm.get("test") or {}
    origin_val = origin_fm.get("val") or {}
    origin_test_top1 = origin_test.get("top1")
    origin_test_top5 = origin_test.get("top5")
    origin_val_top1 = origin_val.get("top1") or _best(
        origin_hist.get("val_accuracy") if origin_hist else None
    )
    origin_val_top5 = origin_val.get("top5")

    yolo_val_map50 = yolo_final["val"]["map50"] if yolo_final and "val" in yolo_final else None
    yolo_test_map50 = yolo_final["test"]["map50"] if yolo_final and "test" in yolo_final else yolo_map50
    yolo_test_map5095 = yolo_final["test"]["map50_95"] if yolo_final and "test" in yolo_final else yolo_map5095

    summary = {
        "vehicle_test_acc": vehicle_test_acc,
        "vehicle_val_acc": vehicle_val_acc,
        "origin_test_top1": origin_test_top1,
        "origin_test_top5": origin_test_top5,
        "origin_val_top1": origin_val_top1,
        "origin_val_top5": origin_val_top5,
        "yolo_test_map50": yolo_test_map50,
        "yolo_test_map5095": yolo_test_map5095,
        "yolo_val_map50": yolo_val_map50,
    }

    return render_template(
        "metrics.html",
        vehicle_hist=vehicle_hist,
        origin_hist=origin_hist,
        yolo_hist=yolo_hist,
        yolo_final=yolo_final,
        summary=summary,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _warm_up() -> None:
    try:
        _init_pipeline()
        logger.info("Pipeline cargado correctamente.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pipeline parcialmente cargado (modelos pendientes): %s", exc)


if __name__ == "__main__":
    _warm_up()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
