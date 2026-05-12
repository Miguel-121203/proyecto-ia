# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Guía de contexto para Claude Code en el proyecto **proyecto-ia-placas**.

## Contexto del proyecto

Proyecto final de la asignatura **Inteligencia Artificial** del programa de Ingeniería de Sistemas en CORHUILA (2026-1). Docente: José Miguel Llanos Mosquera. Sustentación: **19 de mayo de 2026**.

Sistema de visión por computadora que, dada una imagen de un vehículo, ejecuta un pipeline en cascada de modelos de deep learning:

1. Clasifica tipo de vehículo (moto / carro).
2. Detecta la bounding box de la placa.
3. Clasifica el estado/territorio USA de origen de la placa.

**Hook narrativo:** sistema de control de acceso vehicular automatizado para parqueaderos corporativos.

## Stack técnico obligatorio

- **Lenguaje:** Python 3.11
- **Framework web:** Flask (no Django, por peso y simplicidad de servir modelos).
- **Deep learning:** TensorFlow 2.x / Keras (modelos 1 y 3) · Ultralytics YOLOv8 (modelo 2).
- **Frontend:** HTML + Bootstrap 5 + Chart.js.
- **Entrenamiento:** local en macOS Apple Silicon (M4) con `tensorflow-metal` (GPU integrado vía Metal). Ultralytics detecta MPS automáticamente para YOLO.
- **IDE:** VS Code con extensiones Python. No se usa Google Colab.
- **Despliegue:** Hugging Face Spaces con Docker (NO Render, RAM insuficiente).
- **Control de versiones:** Git + GitHub + Git LFS para modelos > 50 MB.

## Arquitectura de modelos

| # | Modelo | Arquitectura | Input | Output |
|---|--------|--------------|-------|--------|
| 1 | `vehicle_classifier` | MobileNetV2 + transfer learning | RGB 224×224 | `{moto, carro}` + confianza |
| 2 | `plate_detector` | **YOLOv8s** (fine-tuned) | RGB 640×640 | Bounding box (x, y, w, h) + confianza |
| 3 | `origin_classifier` | **MobileNetV2** + transfer learning | RGB 300×300 | Estado/territorio USA + confianza |

### Splits estándar (70 / 15 / 15)

- **train** — ajusta pesos.
- **val** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- **test** — holdout intocable; métricas oficiales del informe.

### Regla de flujo

El Modelo 3 **depende** del recorte que produce el Modelo 2. Nunca pasar la imagen completa al clasificador de origen.

## Estructura del repositorio

```
proyecto-ia-placas/
├── run.py                                # ⭐ entry point: auto-entrena faltantes y sirve Flask :7860
├── src/                                  # código de inferencia
│   ├── config.py                         # MODEL_PATHS + InferenceConfig
│   ├── pipeline.py                       # orquestador en cascada
│   ├── vehicle_classifier.py
│   ├── plate_detector.py
│   └── origin_classifier.py
├── app/
│   ├── app.py                            # Flask app
│   ├── templates/
│   │   ├── index.html                    # UI subida de imagen
│   │   └── metrics.html                  # KPIs + Chart.js + matrices
│   └── static/plots/                     # PNG/JSON/CSV generados al entrenar
├── scripts/
│   ├── data/                             # preparación de datos
│   │   ├── split_vehicles.py             # split 70/15/15 con symlinks
│   │   ├── voc_to_yolo.py                # Pascal VOC → YOLO con train/val/test
│   │   ├── combine_plates.py             # gpiosenka 50 + 56 → us_license_plates
│   │   └── add_external_plates.py        # tolgadincer + jonkeegan → train set
│   └── training/                         # entrenamiento
│       ├── train_vehicle.py
│       ├── train_yolo.py
│       └── train_origin.py
├── models/                               # .h5 + .pt + origin_classes.json (Git LFS)
├── data/
│   ├── raw/
│   │   ├── modelo1/{Bike,Car}/           # 4 000 imgs (clasificación vehículo)
│   │   ├── modelo2/{images,labels}/{train,val,test}/  # 26 929 imgs YOLO (detección placa)
│   │   └── modelo3/
│   │       ├── fuentes/                  # 4 datasets crudos (4.3 GB)
│   │       └── us_license_plates/        # combinado: ~26 K imgs · 51 clases
│   └── processed/                        # vehicles_split/, plates_yolo/ (regenerable)
├── Dockerfile                            # HF Spaces
├── requirements.txt
├── README.md
├── CLAUDE.md
└── .gitignore
```

### Entry points

- `python run.py` — entrenamiento completo si faltan modelos + sirve Flask en `:7860`.
- `python run.py --fast` — modo demo (~3-5 min).
- `python app/app.py` — solo sirve (usado en Docker / HF Spaces).
- `python scripts/training/train_<modelo>.py [--fast]` — entrenar uno puntual.
- `python scripts/data/<script>.py` — preparar dataset.

## Datasets

### Modelo 1 — Clasificación moto/carro
- **Directorio:** `data/raw/modelo1/{Bike,Car}/` (Car-Bike-Dataset · 2 000 + 2 000).
- **Split:** `python scripts/data/split_vehicles.py` genera `data/processed/vehicles_split/{train,val,test}/{Bike,Car}/` con symlinks (70/15/15, semilla 42).
- Labels alfabéticos de `flow_from_directory`: `Bike → 0 (moto)`, `Car → 1 (carro)`.

### Modelo 2 — Detección de placa
- **Directorio:** `data/raw/modelo2/{images,labels}/{train,val,test}/` (Kaggle `fareselmenshawii/large-license-plate-dataset` · **26 929 imgs ya en formato YOLO**).
- Splits incluidos: train **25 470** · val **1 073** · test **386**.
- `dataset.yaml` se genera automáticamente en `data/raw/modelo2/dataset.yaml` la primera vez que corres `train_yolo.py`.
- Si más adelante se usa otro dataset en formato Pascal VOC (`annotations/*.xml`), el script lo detecta y llama a `voc_to_yolo.py` para convertirlo a `data/processed/plates_yolo/`.
- Respaldo del dataset viejo (433 imgs VOC, `andrewmvd/car-license-plate-detection`) en `~/Downloads/archivee/`.

### Modelo 3 — Clasificador de estado USA
- **Directorio combinado:** `data/raw/modelo3/us_license_plates/{train,valid,test}/<CLASE>/`.
- **Fuentes crudas:** `data/raw/modelo3/fuentes/` (~4.3 GB):
  - `gpiosenka_50_estados` (gpiosenka/us-license-plates-image-classification, 50 clases).
  - `gpiosenka_56_clases_territorios` (versión con territorios).
  - `tolgadincer_50_estados` (~4 463 imgs train).
  - `jonkeegan_51_estados_foia` (FOIA, ~8 239 imgs train, GIFs animados excluidos).
- Estado actual tras `combine_plates.py + add_external_plates.py`: **train ~26 565 imgs · valid 530 · test 530**.
- **51 clases** finales: 50 estados + WASHINGTON DC. Territorios (American Samoa, CNMI, Guam, Puerto Rico, US Virgin Islands) excluidos por decisión de diseño.
- Dos typos preservados del dataset original (consistencia entre train e inference vía `models/origin_classes.json`):
  - `HAWAI` (sin segunda I).
  - `MISSIPPI` (sin doble S).

## Hiperparámetros

### Modelo 1 — `train_vehicle.py`
| Parámetro | Valor | Notas |
|---|---|---|
| input_size | 224×224 | Default MobileNetV2 |
| batch_size | 32 | |
| epochs head | 6 | Backbone congelado |
| epochs fine-tuning | 3 | Últimas 20 capas |
| lr head / ft | 1e-3 / 1e-5 | |
| EarlyStopping patience | 3 | |

### Modelo 2 — `train_yolo.py`
| Parámetro | Valor | Notas |
|---|---|---|
| modelo base | `yolov8s.pt` | Upgrade desde nano (mAP +3-5 pts) |
| imgsz | 640 | |
| batch | 16 | |
| epochs | 4 | ~3h en M4 con 100% data; con 25K imgs/época hay mucha variedad |
| patience | 1 | Corta inmediatamente si no mejora |
| device | `mps` | Apple Silicon |

### Modelo 3 — `train_origin.py`
| Parámetro | Valor | Notas |
|---|---|---|
| arquitectura | MobileNetV2 | Config que produjo el modelo de referencia (87.7% test) |
| input_size | 300×300 | Personalizado, mejor que el default 224 para placas |
| batch_size | 32 | |
| epochs head | 15 | |
| epochs fine-tuning | 50 | + 15 adicionales con `--resume-finetune` ≈ 80 totales |
| lr head / ft | 1e-3 / 3e-5 (1e-5 en `--resume-finetune`) | |
| capas descongeladas | 20 (40 en `--resume-finetune`) | |
| label smoothing | 0.1 | |
| EarlyStopping patience | 6 head / 8 ft | |

## PipelineResult

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

Los modelos se cargan **una sola vez** al instanciar `VehiclePipeline`, no en cada `predict()`.

## Flask app

- `GET /` — formulario HTML para subir imagen.
- `POST /predict` — recibe imagen (multipart), devuelve JSON.
- `GET /metrics` — KPIs + curvas Chart.js + matrices de confusión (val + test).
- `GET /health` — healthcheck.
- Límite de subida: 10 MB. Formatos: JPG / PNG / WEBP.

## Las gráficas en /metrics

Cada `scripts/training/train_*.py` deja sus artefactos en `app/static/plots/`:

1. **Modelo 1** — `history_vehicle.json`, `vehicle_confusion_matrix.png` (val), `vehicle_confusion_matrix_test.png` (holdout), `training_curves_vehicle.png`.
2. **Modelo 2** — `yolo_results.csv`, `yolo_pr_curve.png`, `yolo_val_preds.jpg`, `yolo_metrics.json` (val + test mAP).
3. **Modelo 3** — `history_origin.json`, `origin_confusion_matrix.png` (val), `origin_confusion_matrix_test.png` (holdout, métrica oficial).

## Despliegue

- **Plataforma:** Hugging Face Spaces (Docker, 16 GB RAM).
- **Dockerfile:** `python:3.11-slim`, `EXPOSE 7860`, `CMD ["python", "app/app.py"]`.
- **Flujo:** entrenar local → subir modelos con Git LFS → HF Spaces sirve solo (no reentrena).
- **Variable:** `MODELS_DIR=/app/models` en el contenedor.

## Estado actual (29 abr 2026)

- ✅ Datasets reorganizados dentro del proyecto en `data/raw/modelo{1,2,3}/`. Respaldo intacto en `~/Downloads/`.
- ✅ `us_license_plates/` regenerado: train 26 565 · valid 530 · test 530.
- ⚠️ **Pendiente reentrenar los 3 modelos** tras estos cambios:
  - Splits cambiaron a 70/15/15 con holdout test independiente.
  - `plate_detector`: YOLOv8n → YOLOv8s.
  - `origin_classifier`: MobileNetV2 300×300 → EfficientNetV2B0 240×240.

### Pendientes (orden sugerido)

1. `python run.py` para reentrenar los 3 modelos con los nuevos hiperparámetros y splits.
2. Validar pipeline end-to-end con imágenes reales.
3. Validar `Dockerfile` (`docker build . && docker run -p 7860:7860`).
4. `git init` + Git LFS + push a GitHub.
5. Desplegar en Hugging Face Spaces.
6. Informe PDF (10+ págs APA 7).
7. Video demo backup.

## Convenciones de código

- **Estilo:** PEP 8, formateado con Black (línea 100).
- **Typing:** type hints en todas las funciones públicas.
- **Docstrings:** formato Google style.
- **Logging:** usar `logging` stdlib, no `print` (excepción: scripts de entrenamiento usan `print` para streaming en vivo).
- **No hardcodear paths:** usar `pathlib.Path` y `src/config.py`.

## Riesgos conocidos

| Riesgo | Mitigación |
|--------|------------|
| Modelos exceden RAM de HF Spaces | M4 16 GB · HF Spaces 16 GB. Si excede, cuantizar a TFLite. |
| `origin_classifier` Top-1 limitado en estados similares | Reportar Top-5 + matriz de confusión. EfficientNetV2B0 debería mejorar baseline. |
| Cold start lento en HF Spaces | Pre-cargar modelos al iniciar el contenedor (vía `_warm_up`). |
| Falla de internet en sustentación | Grabar video demo + tener app corriendo en localhost. |
| `tensorflow-metal` inestable | `export TF_DISABLE_METAL=1` como workaround. |

## Reglas para Claude Code

1. **Siempre verificar el estado actual antes de crear archivos.** Usar `ls` al inicio de cada sesión.
2. **No reentrenar modelos sin consultar.** Si el peso existe en `models/`, usarlo.
3. **No hardcodear paths.** Usar `pathlib.Path` y `src/config.py`.
4. **Rápido sobre robusto.** Proyecto académico con deadline, no producción.
5. **No crear tests unitarios para código de entrenamiento.**
6. **Commits atómicos** en español, formato `<scope>: <descripción>`.
7. **Actualizar este CLAUDE.md** al cerrar una sesión relevante.

## Notas adicionales

- **Contexto del desarrollador:** Miguel Ángel — backend senior en Tenpo y Bancolombia. No es especialista en ML. Explicar decisiones de ML cuando sean no obvias; asumir dominio total de Python, Docker, CI/CD y Git.
- **Sin over-engineering.** Pipeline lineal académico, no arquitectura hexagonal.
- **Idioma:** código en inglés. Documentación y commits en español.

---

*Última actualización: 29 de abril de 2026 — reorganización scripts/{data,training}, splits 70/15/15, upgrade YOLOv8s + EfficientNetV2B0.*
