# Guía Completa del Pipeline LSM-Core

Esta guía documenta el flujo completo desde la descarga de datos hasta el entrenamiento y visualización.

---

## 🚀 Flujo Rápido

```bash
# 1. Descargar videos nuevos
python -m src.utils.download_videos

# 2. Preprocesar videos
python -m src.extraction.preprocessor

# 3. Inspeccionar datos (opcional)
python -m src.data.inspect_processed

# 4. Entrenar modelo
python -m src.training.train

# 5. Ver resultados en MLflow
mlflow ui --backend-store-uri "sqlite:///experiments/mlruns/mlflow.db"
```

---

## 📥 1. Descarga de Videos

### Incremental (solo nuevos)
```bash
python -m src.utils.download_videos
```

### Completo (todos)
```bash
python -m src.utils.download_videos --full
```

**Ubicación:** `dataset/raw/{clase}/*.mp4`

---

## 🔄 2. Preprocesamiento

### Incremental (solo sin procesar)
```bash
python -m src.extraction.preprocessor
```

### Completo (reprocesar todo)
```bash
python -m src.extraction.preprocessor --full
```

**Pipeline:**
1. RTMPose → 133 keypoints
2. Filtro confianza (< 0.5 → 0,0)
3. OneEuroFilter suavizado
4. Normalización centrada en caderas

**Salida:** `dataset/processed/{clase}/*.npy`

---

## 🔍 3. Inspección de Datos

```bash
python -m src.data.inspect_processed
```

**Genera:**
- Distribución de clases
- Class weights para training
- Histograma de longitud de videos
- Alertas de archivos corruptos

---

## 🧠 4. Entrenamiento

```bash
python -m src.training.train
```

**Configuración en `src/config/settings.py`:**
```python
# Modelo
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.4

# Training
EPOCHS = 150
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1

# Pesos por clase (balance)
CLASS_WEIGHTS = [1.2515, 1.2294, 1.3933, 1.1117, 0.5649]

# Pesos por región del cuerpo
FEATURE_WEIGHTS = {
    'body': 1.0,
    'feet': 0.3,
    'face': 0.1,
    'left_hand': 2.5,
    'right_hand': 2.5
}
```

---

## 📊 5. MLflow UI

```bash
cd "C:\Users\juana\Documents\Trabajo 2025\lsm-translator-core"
mlflow ui --backend-store-uri "sqlite:///experiments/mlruns/mlflow.db"
```

Luego abrir: **http://localhost:5000**

---

## 🎥 6. Inferencia

### Validar con videos del dataset
```bash
python -m src.inference.video_demo
```

### Demo en tiempo real (iPad/DroidCam)
```bash
python -m src.inference.ipad_demo
```

---

## 📁 Estructura de Archivos

```
lsm-translator-core/
├── dataset/
│   ├── .last_sync              # Timestamp último sync
│   ├── raw/                    # Videos .mp4
│   └── processed/              # Tensores .npy
├── experiments/
│   └── mlruns/
│       ├── mlflow.db           # Base de datos MLflow
│       ├── best_model.pth      # Mejor modelo
│       ├── confusion_matrix.png
│       └── training_curves.png
├── src/
│   ├── config/settings.py      # CONFIGURACIÓN CENTRAL
│   ├── extraction/
│   ├── training/
│   ├── inference/
│   └── models/
└── docs/
```

---

## ⚙️ Parámetros Clave

| Parámetro | Archivo | Valor |
|-----------|---------|-------|
| `CONFIDENCE_THRESHOLD` | settings.py | 0.50 |
| `D_MODEL` | settings.py | 128 |
| `N_LAYERS` | settings.py | 3 |
| `DROPOUT` | settings.py | 0.4 |
| `LEARNING_RATE` | settings.py | 3e-4 |
| `EPOCHS` | settings.py | 150 |
| `CLASS_WEIGHTS` | settings.py | [1.25, 1.23, 1.39, 1.11, 0.56] |
| `FEATURE_WEIGHTS.face` | settings.py | 0.1 |
| `FEATURE_WEIGHTS.hands` | settings.py | 2.5 |
