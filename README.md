# proyecto-ia-placas

Sistema de visión por computadora que, dada la imagen de un vehículo, ejecuta un pipeline en cascada de modelos de deep learning:

1. Clasifica el tipo de vehículo (moto / carro).
2. Detecta la bounding box de la placa.
3. Clasifica el estado / territorio USA de origen de la placa.

Proyecto final de la asignatura **Inteligencia Artificial** — CORHUILA, 2026-1. Sustentación: **19 de mayo de 2026**.

---

## Stack

- **Python** 3.11
- **MobileNetV2** (modelo 1) · **YOLOv8s** (modelo 2) · **EfficientNetV2B0** (modelo 3)
- **TensorFlow / Keras** + **Ultralytics**
- **Flask** + **Bootstrap 5** + **Chart.js**
- **Docker** → **Hugging Face Spaces**

## Estructura

```
proyecto-ia-placas/
├── run.py                   # entry point (entrena lo que falte y sirve Flask :7860)
├── src/                     # código de inferencia (pipeline)
│   ├── config.py
│   ├── pipeline.py
│   ├── vehicle_classifier.py
│   ├── plate_detector.py
│   └── origin_classifier.py
├── app/
│   ├── app.py               # Flask app
│   ├── templates/           # index.html · metrics.html
│   └── static/plots/        # PNG/JSON generados al entrenar
├── scripts/
│   ├── data/                # preparación de datasets
│   │   ├── split_vehicles.py
│   │   ├── voc_to_yolo.py
│   │   ├── combine_plates.py
│   │   └── add_external_plates.py
│   └── training/            # entrenamiento de los 3 modelos
│       ├── train_vehicle.py
│       ├── train_yolo.py
│       └── train_origin.py
├── models/                  # pesos finales (Git LFS)
├── data/                    # datasets reales (gitignored)
├── Dockerfile
├── requirements.txt
└── CLAUDE.md
```

## Instalación local (macOS Apple Silicon)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Verificar Metal GPU del M4
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Preparación de datasets

Los datasets viven dentro del proyecto, organizados por modelo:

| Modelo | Carpeta | Contenido |
|---|---|---|
| 1 vehículo | `data/raw/modelo1/{Bike,Car}/` | Car-Bike-Dataset · 4 000 imgs |
| 2 placa | `data/raw/modelo2/{images,labels}/{train,val,test}/` | Kaggle YOLO `fareselmenshawii/large-license-plate-dataset` · 26 929 imgs |
| 3 estado USA | `data/raw/modelo3/us_license_plates/{train,valid,test}/` | combinado · ~26 K imgs train (51 clases) |
| 3 fuentes | `data/raw/modelo3/fuentes/` | datasets crudos (gpiosenka × 2, tolgadincer, jonkeegan) |

> Los datasets se respaldan en `~/Downloads/` (Car-Bike-Dataset, archivee). El proyecto trabaja sobre las copias en `data/raw/modelo*/`.

### Splits 70 / 15 / 15 (train / val / test)

```bash
# Modelo 1 — split físico con symlinks
python scripts/data/split_vehicles.py

# Modelo 2 — el dataset ya viene en YOLO; el dataset.yaml se genera automáticamente
# (Si bajaras un dataset VOC en su lugar, voc_to_yolo.py lo convierte:)
# python scripts/data/voc_to_yolo.py --voc-dir data/raw/modelo2 \
#   --out-dir data/processed/plates_yolo --classes licence \
#   --val-split 0.15 --test-split 0.15

# Modelo 3 — el dataset ya viene partido (train/valid/test)
python scripts/data/combine_plates.py
python scripts/data/add_external_plates.py    # opcional
```

## Ejecución

```bash
# Entrena todo lo que falte y arranca Flask
python run.py
python run.py --fast     # demo rápido (~3 min)

# O entrenar un modelo puntual
python scripts/training/train_vehicle.py
python scripts/training/train_yolo.py
python scripts/training/train_origin.py [--fast | --reset | --resume-finetune]

# Solo servir (Docker / HF Spaces)
python app/app.py
```

Endpoints:

- `GET /` formulario de subida.
- `POST /predict` recibe la imagen y devuelve JSON.
- `GET /metrics` curvas + matrices de confusión + KPIs.
- `GET /health` healthcheck.

## Despliegue (Hugging Face Spaces)

```bash
git remote add hf https://huggingface.co/spaces/<usuario>/proyecto-ia-placas
git push hf main
```

El `Dockerfile` usa `python:3.11-slim`, expone `:7860` y arranca `python app/app.py` (no reentrena).

## Autoría

Miguel Ángel — Ingeniería de Sistemas, CORHUILA. Docente: José Miguel Llanos Mosquera.
