# proyecto-ia-placas

Sistema de visión por computadora que, dada la imagen de un vehículo, ejecuta un **pipeline en cascada de tres modelos de deep learning**:

1. **Clasifica el tipo de vehículo** (moto / carro).
2. **Detecta la placa** (bounding box).
3. **Clasifica el estado/territorio USA** al que pertenece la placa.

Hook narrativo: *control de acceso vehicular automatizado para parqueaderos corporativos*.

> Proyecto final de la asignatura **Inteligencia Artificial** — Ingeniería de Sistemas, CORHUILA · 2026-1.
> Docente: José Miguel Llanos Mosquera · Sustentación: 19 de mayo de 2026.

---

## Tabla de contenidos

- [Demo rápida](#demo-rápida)
- [Arquitectura](#arquitectura)
- [Stack técnico](#stack-técnico)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso](#uso)
- [Entrenamiento](#entrenamiento)
- [Datasets](#datasets)
- [Hiperparámetros](#hiperparámetros)
- [API HTTP](#api-http)
- [Despliegue (Docker / HF Spaces)](#despliegue-docker--hf-spaces)
- [Convenciones de código](#convenciones-de-código)
- [Riesgos conocidos](#riesgos-conocidos)
- [Licencia y autores](#licencia-y-autores)

---

## Demo rápida

```bash
# 1. Clonar el repo (incluye modelos vía Git LFS)
git clone https://github.com/Miguel-121203/proyecto-ia.git
cd proyecto-ia
git lfs pull                                # descarga los .h5 y .pt

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Arrancar la app web
python app/app.py                           # http://localhost:7860
```

No requiere reentrenar: los pesos viajan versionados en Git LFS.

---

## Arquitectura

Pipeline lineal donde la salida del **Modelo 2** alimenta directamente al **Modelo 3**. Los modelos se cargan **una sola vez** al instanciar `VehiclePipeline`, nunca por petición.

```
        ┌──────────────────────┐
imagen →│  M1: vehicle_class.  │── moto / carro + confianza
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  M2: plate_detector  │── bbox + confianza
        └──────────┬───────────┘
                   │ recorte de la placa
                   ▼
        ┌──────────────────────┐
        │  M3: origin_class.   │── estado USA + confianza (Top-1 / Top-5)
        └──────────────────────┘
```

| # | Modelo                | Arquitectura                       | Input       | Output                          |
|---|-----------------------|------------------------------------|-------------|---------------------------------|
| 1 | `vehicle_classifier`  | MobileNetV2 + transfer learning    | RGB 224×224 | `{moto, carro}` + confianza     |
| 2 | `plate_detector`      | YOLOv8s (fine-tuned)               | RGB 640×640 | Bounding box `(x, y, w, h)` + conf. |
| 3 | `origin_classifier`   | MobileNetV2 + transfer learning    | RGB 300×300 | Estado/territorio USA + conf.   |

Resultado tipado de la inferencia:

```python
@dataclass
class PipelineResult:
    vehicle_type: str
    vehicle_confidence: float
    plate_bbox: tuple[int, int, int, int] | None = None
    plate_confidence: float | None = None
    origin: str | None = None
    origin_confidence: float | None = None
```

---

## Stack técnico

| Capa              | Tecnología                                                                 |
|-------------------|----------------------------------------------------------------------------|
| Lenguaje          | Python 3.11                                                                |
| Web               | Flask 3 + Werkzeug                                                         |
| Deep learning     | TensorFlow 2.16 / Keras (M1, M3) · Ultralytics YOLOv8 (M2)                 |
| Aceleración       | `tensorflow-metal` en Apple Silicon · MPS automático para Ultralytics      |
| Frontend          | HTML + Bootstrap 5 + Chart.js                                              |
| Visualización     | matplotlib · scikit-learn (matrices de confusión)                          |
| Despliegue        | Docker · Hugging Face Spaces (16 GB RAM)                                   |
| Versionado pesos  | Git LFS (modelos > 50 MB)                                                  |

---

## Estructura del repositorio

```
proyecto-ia/
├── run.py                       # ⭐ entry point: entrena lo que falte y sirve Flask :7860
├── src/                         # código de inferencia
│   ├── config.py                # MODEL_PATHS + InferenceConfig
│   ├── pipeline.py              # orquestador en cascada
│   ├── vehicle_classifier.py
│   ├── plate_detector.py
│   └── origin_classifier.py
├── app/
│   ├── app.py                   # Flask app
│   ├── templates/
│   │   ├── index.html           # UI subida de imagen
│   │   └── metrics.html         # KPIs + Chart.js + matrices
│   └── static/plots/            # PNG / JSON / CSV generados al entrenar
├── scripts/
│   ├── data/                    # preparación de datos
│   │   ├── split_vehicles.py
│   │   ├── voc_to_yolo.py
│   │   ├── combine_plates.py
│   │   └── add_external_plates.py
│   └── training/                # entrenamiento de los 3 modelos
│       ├── train_vehicle.py
│       ├── train_yolo.py
│       └── train_origin.py
├── models/                      # .h5 + .pt + origin_classes.json (Git LFS)
├── Dockerfile                   # imagen para HF Spaces
├── requirements.txt
├── CLAUDE.md                    # guía interna de contexto
└── README.md
```

---

## Instalación

### Requisitos

- Python **3.11**
- `git` y `git-lfs`
- ~150 MB libres (pesos) · ~5 GB extra si vas a reentrenar (datasets)

### Pasos

```bash
git clone https://github.com/Miguel-121203/proyecto-ia.git
cd proyecto-ia

# Pesos versionados con LFS
git lfs install
git lfs pull

# Entorno virtual (recomendado)
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> En macOS Apple Silicon, `tensorflow-metal` se instala automáticamente y habilita la GPU integrada. Si causa inestabilidad: `export TF_DISABLE_METAL=1`.

---

## Uso

### Servir la app

```bash
python app/app.py            # http://localhost:7860
```

### Entry point todo-en-uno

```bash
python run.py                # entrena los modelos faltantes y luego sirve la app
python run.py --fast         # modo demo: entrenamiento mínimo en ~3-5 min
```

Si los pesos ya existen en `models/`, el entrenamiento se omite.

### Inferencia programática

```python
import numpy as np
from PIL import Image
from src.pipeline import VehiclePipeline

pipeline = VehiclePipeline()                       # carga los 3 modelos una sola vez
img = np.array(Image.open("carro.jpg").convert("RGB"))
result = pipeline.predict(img)

print(result.vehicle_type, result.vehicle_confidence)
print(result.plate_bbox, result.plate_confidence)
print(result.origin, result.origin_confidence)
```

---

## Entrenamiento

Cada modelo se entrena con su propio script. Los pesos se guardan en `models/` y las métricas/curvas en `app/static/plots/`.

```bash
python scripts/training/train_vehicle.py            # MobileNetV2 — moto/carro
python scripts/training/train_yolo.py               # YOLOv8s — detección de placa
python scripts/training/train_origin.py             # MobileNetV2 — estado USA
```

Bandera `--fast` disponible en todos para iteración rápida durante desarrollo.

### Splits estándar (70 / 15 / 15)

- **train** — ajuste de pesos.
- **val** — `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
- **test** — holdout intocable; métricas oficiales del informe.

---

## Datasets

### Modelo 1 — Vehículo (moto / carro)

- Origen: *Car-Bike-Dataset* (2 000 + 2 000 imágenes).
- Ruta: `data/raw/modelo1/{Bike,Car}/`.
- Split con symlinks: `python scripts/data/split_vehicles.py` → `data/processed/vehicles_split/{train,val,test}/`.
- Labels alfabéticos de `flow_from_directory`: `Bike → 0 (moto)`, `Car → 1 (carro)`.

### Modelo 2 — Detección de placa

- Origen: Kaggle `fareselmenshawii/large-license-plate-dataset` (**26 929 imgs** ya en formato YOLO).
- Splits incluidos: train **25 470** · val **1 073** · test **386**.
- `dataset.yaml` se genera automáticamente la primera vez que corres `train_yolo.py`.
- Compatible con datasets Pascal VOC vía `scripts/data/voc_to_yolo.py`.

### Modelo 3 — Estado USA

- Combinación de 4 fuentes: `gpiosenka_50_estados`, `gpiosenka_56_clases_territorios`, `tolgadincer_50_estados`, `jonkeegan_51_estados_foia`.
- Tras `combine_plates.py + add_external_plates.py`: **train ~26 565 · valid 530 · test 530**.
- **51 clases** = 50 estados + Washington DC (territorios excluidos por decisión de diseño).
- Dos *typos* conservados del dataset original para consistencia train/inference (vía `models/origin_classes.json`): `HAWAI` y `MISSIPPI`.

---

## Hiperparámetros

### Modelo 1 — `train_vehicle.py`

| Parámetro                | Valor       |
|--------------------------|-------------|
| Input                    | 224×224     |
| Batch size               | 32          |
| Epochs head / fine-tune  | 6 / 3       |
| LR head / fine-tune      | 1e-3 / 1e-5 |
| EarlyStopping patience   | 3           |

### Modelo 2 — `train_yolo.py`

| Parámetro     | Valor          |
|---------------|----------------|
| Modelo base   | `yolov8s.pt`   |
| imgsz         | 640            |
| Batch         | 16             |
| Epochs        | 4              |
| Patience      | 1              |
| Device        | `mps` (Apple)  |

### Modelo 3 — `train_origin.py`

| Parámetro                        | Valor                              |
|----------------------------------|------------------------------------|
| Arquitectura                     | MobileNetV2                        |
| Input                            | 300×300                            |
| Batch size                       | 32                                 |
| Epochs head / fine-tune          | 15 / 50 (+15 con `--resume-finetune`) |
| LR head / fine-tune              | 1e-3 / 3e-5 (1e-5 en resume)       |
| Capas descongeladas              | 20 (40 en resume)                  |
| Label smoothing                  | 0.1                                |
| EarlyStopping patience head / ft | 6 / 8                              |

---

## API HTTP

Flask sirve por defecto en `:7860`. Límite de subida: **10 MB**. Formatos: `jpg`, `jpeg`, `png`, `webp`.

| Método | Ruta        | Descripción                                                   |
|--------|-------------|---------------------------------------------------------------|
| GET    | `/`         | Formulario HTML para subir imagen.                             |
| POST   | `/predict`  | Recibe `multipart/form-data` con campo `image`. Devuelve JSON. |
| GET    | `/metrics`  | KPIs + curvas Chart.js + matrices de confusión (val + test).   |
| GET    | `/health`   | Healthcheck → `{"status": "ok"}`.                              |

### Ejemplo — `POST /predict`

```bash
curl -X POST http://localhost:7860/predict \
     -F "image=@carro.jpg"
```

```json
{
  "vehicle_type": "carro",
  "vehicle_confidence": 0.987,
  "plate_bbox": [120, 340, 285, 80],
  "plate_confidence": 0.94,
  "origin": "CALIFORNIA",
  "origin_confidence": 0.78
}
```

Códigos de error: `400` (input inválido) · `503` (modelos no entrenados) · `500` (fallo interno).

---

## Despliegue (Docker / HF Spaces)

El `Dockerfile` está pensado para **Hugging Face Spaces** (16 GB RAM). Imagen base `python:3.11-slim`, expone el puerto **7860**.

```bash
docker build -t proyecto-ia .
docker run -p 7860:7860 proyecto-ia
# → http://localhost:7860
```

Flujo recomendado:

1. Entrenar **localmente** los 3 modelos.
2. Subir los pesos al repo vía **Git LFS**.
3. HF Spaces hace `git clone` y arranca con `python app/app.py` — **no reentrena**.

Variables relevantes:

- `MODELS_DIR=/app/models` (configurable, por defecto en el contenedor).
- `PORT=7860`.

---

## Convenciones de código

- **Estilo:** PEP 8, formateado con `black` (línea 100) e `isort`.
- **Typing:** type hints en toda función pública.
- **Docstrings:** Google style.
- **Logging:** `logging` stdlib (excepción: scripts de entrenamiento usan `print` para streaming).
- **Paths:** siempre vía `pathlib.Path` y `src/config.py`. Nunca hardcodear.
- **Idioma:** código en inglés, documentación y commits en español, formato `<scope>: <descripción>`.

---

## Riesgos conocidos

| Riesgo                                              | Mitigación                                                  |
|-----------------------------------------------------|-------------------------------------------------------------|
| Modelos exceden RAM de HF Spaces                    | Cuantizar a TFLite si excede los 16 GB.                     |
| Top-1 limitado en estados visualmente similares     | Reportar Top-5 + matriz de confusión.                       |
| Cold start lento en HF Spaces                       | Pre-carga de modelos vía `_warm_up()` al iniciar.           |
| Falla de internet en sustentación                   | Backup: video demo + app corriendo en localhost.            |
| `tensorflow-metal` inestable                        | `export TF_DISABLE_METAL=1`.                                |

---

## Licencia y autores

- **Autores:** Miguel Ángel Beltran Bohorquez, Nicolas Rivera Aroca, Huber Andres Parra Molina — Ingeniería de Sistemas, CORHUILA 2026-1.
- **Docente:** José Miguel Llanos Mosquera.
- **Uso:** académico. Datasets de terceros conservan sus licencias originales (Kaggle / gpiosenka / tolgadincer / jonkeegan FOIA).
