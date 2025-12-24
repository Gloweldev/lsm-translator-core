# Guía de Pipeline de Datos

Esta guía explica cómo descargar y procesar videos para entrenar el modelo LSM.

---

## 📥 Descarga de Videos

### Descargar solo videos nuevos (Incremental)
```bash
python -m src.utils.download_videos
```
- Usa el archivo `.last_sync` para descargar solo videos creados después del último sync
- Omite videos que ya existen localmente

### Descargar todos los videos (Completo)
```bash
python -m src.utils.download_videos --full
```
- Ignora el `.last_sync`
- Descarga todos los videos del API
- Útil para reset completo del dataset

---

## 🔄 Preprocesamiento

### Procesar solo videos nuevos (Incremental)
```bash
python -m src.extraction.preprocessor
```
- Solo procesa videos que no tienen un `.npy` correspondiente
- Rápido para actualizaciones incrementales

### Reprocesar todos los videos (Completo)
```bash
python -m src.extraction.preprocessor --full
```
- Elimina y regenera todos los `.npy`
- Útil después de cambiar parámetros de preprocesamiento

---

## 🚀 Flujo Completo de Actualización

### Caso 1: Nuevos videos agregados al servidor
```bash
# 1. Descargar solo los nuevos
python -m src.utils.download_videos

# 2. Procesar solo los nuevos
python -m src.extraction.preprocessor

# 3. (Opcional) Reentrenar el modelo
python -m src.training.train
```

### Caso 2: Reset completo del dataset
```bash
# 1. Descargar todo
python -m src.utils.download_videos --full

# 2. Reprocesar todo
python -m src.extraction.preprocessor --full

# 3. Reentrenar
python -m src.training.train
```

### Caso 3: Cambio en parámetros de preprocesamiento
```bash
# Solo reprocesar (no descargar)
python -m src.extraction.preprocessor --full

# Reentrenar
python -m src.training.train
```

---

## 📁 Estructura de Archivos

```
dataset/
├── .last_sync              # Timestamp del último sync
├── raw/                    # Videos originales (.mp4)
│   ├── a/
│   ├── b/
│   ├── hola/
│   └── nada/
└── processed/              # Tensores procesados (.npy)
    ├── a/
    ├── b/
    ├── hola/
    └── nada/
```

---

## ⚙️ Parámetros del Preprocesador

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.50 | Keypoints con score < 0.5 → (0,0) |
| `FILTER_MIN_CUTOFF` | 0.1 | OneEuroFilter suavizado |
| `FILTER_BETA` | 0.009 | OneEuroFilter velocidad |
| RTMPose Model | wholebody-384x288 | 133 keypoints |
| Output Dim | 266 | 133 × 2 (x, y) |

---

## 🔍 Verificación de Datos

### Inspeccionar tensores procesados
```bash
python -m src.data.inspect_processed
```

Genera:
- Distribución de clases
- Class weights para training
- Histograma de longitud de videos
- Alertas de archivos corruptos
