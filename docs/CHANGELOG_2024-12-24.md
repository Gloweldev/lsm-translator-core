# Changelog - Sesión 24 Diciembre 2024

## Commit sugerido:
```
feat(inference): FSM, veto físico, gráfica temporal y temporal augmentation

- Add FSM (Finite State Machine) para detección estable de palabras
- Add veto físico basado en posición de manos (elimina fantasma del buffer)
- Add gráfica temporal en tiempo real para ipad_demo
- Add temporal data augmentation en entrenamiento (random crop, time warp)
- Add análisis temporal forense para diagnóstico de buffer
- Fix OneEuroFilter smoothing en demos de inferencia
- Fix imports CLASSES -> CLASS_NAMES
- Log source code como artifacts en MLflow
```

---

## 📋 Resumen de Cambios

Esta sesión se enfocó en **eliminar el problema de inercia del buffer** y mejorar la experiencia de inferencia en tiempo real.

---

## 1. 🎯 Máquina de Estados Finita (FSM)

**Archivo:** `src/inference/ipad_demo.py`

### Problema Resuelto
La predicción "parpadeaba" repetidamente mientras la seña estaba activa, mostrando la misma palabra múltiples veces.

### Solución
Implementé una FSM con dos estados:

| Estado | Descripción |
|--------|-------------|
| `IDLE` | Esperando seña |
| `ACTIVE` | Seña detectada |

### Transiciones
```
IDLE → ACTIVE:  confianza > 85% AND clase ≠ "nada"
ACTIVE → IDLE:  confianza < 50% OR clase == "nada"  
ACTIVE → ACTIVE: Hot-Swap a nueva clase (permite frases)
```

### Nueva Clase `WordFSM`
- `update(prediction, confidence)` → Retorna nueva palabra (solo cuando cambia)
- `get_current_word()` → Palabra activa actual
- `get_history()` → Últimas 5 palabras detectadas

---

## 2. 🚫 Veto Físico (Posición de Descanso)

**Archivo:** `src/inference/ipad_demo.py`

### Problema Resuelto
El "fantasma del buffer" - cuando el usuario baja las manos, el modelo sigue prediciendo la seña anterior por la inercia del buffer rodante.

### Solución
Nueva función `is_pose_active()` que detecta posición de descanso:

```python
def is_pose_active(raw_keypoints, frame_height):
    """
    Retorna False si AMBAS muñecas están por debajo de las caderas.
    Incluye margen de 8% para permitir señas al ombligo.
    """
```

### Índices usados (COCO-WholeBody)
- Muñeca izquierda: `9`
- Muñeca derecha: `10`
- Cadera izquierda: `11`
- Cadera derecha: `12`

### Acciones cuando Veto activo
1. ❌ NO ejecuta el Transformer
2. 🗑️ **Limpia el buffer inmediatamente** (mata la memoria)
3. 📊 Limpia historial de probabilidades
4. 🔴 Muestra "VETO (manos abajo)" en rojo

### Parámetro configurable
```python
POSE_MARGIN = 0.08  # 8% de altura - permite señas bajas
```

---

## 3. 📊 Gráfica Temporal en Tiempo Real

**Archivo:** `src/inference/ipad_demo.py`

### Característica Nueva
La demo ahora muestra una gráfica en tiempo real a la derecha del video que visualiza las probabilidades del Transformer frame por frame.

### Layout
```
[  VIDEO  ] [  GRÁFICA  ]
  Vertical    400px
   720px
```

### Elementos de la gráfica
- **Línea gruesa** → Clase activa/detectada
- **Línea punteada gris** → Clase "nada"
- **Línea roja horizontal** → Umbral de trigger (85%)
- **Fondo negro** con tema oscuro

### Historial de probabilidades
```python
MAX_GRAPH_POINTS = 150  # Últimos 150 frames
```

---

## 4. 🎬 Video Vertical (Portrait Mode)

**Archivo:** `src/inference/ipad_demo.py`

### Característica Nueva
Auto-rotación del video cuando viene en modo horizontal:

```python
if w > h:
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
```

Escala automática a `TARGET_HEIGHT = 720px`.

---

## 5. 🔄 OneEuroFilter en Demos de Inferencia

**Archivos:** `src/inference/ipad_demo.py`, `src/inference/video_demo.py`

### Problema Resuelto
Los demos de inferencia NO aplicaban el filtro OneEuroFilter que sí se usa en el preprocessing de entrenamiento, causando ruido en las predicciones.

### Solución
Añadí el mismo pipeline de smoothing:

```python
from src.utils.smoothing import OneEuroFilter

# En extract_keypoints()
if self.smoothers is None:
    self.smoothers = [OneEuroFilter(t0=t, x0=raw_keypoints[i],
                      min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
                      for i in range(INPUT_DIM)]
```

Parámetros importados de `settings.py`:
- `FILTER_MIN_CUTOFF = 0.1`
- `FILTER_BETA = 0.009`

---

## 6. 📈 Temporal Data Augmentation

**Archivo:** `src/training/train.py`

### Problema Resuelto
"Bias de Retorno al Reposo" - el modelo solo detectaba la seña cuando las manos bajaban al final del video.

### Solución: Nuevos métodos en `LSMDataset`

#### `_apply_temporal_augmentation(seq)` 
```python
# 1. RANDOM CROP (50% prob) - CRÍTICO
crop_ratio = uniform(0.7, 1.0)  # 70-100% del original
start_idx = random  # Punto de inicio aleatorio

# 2. TIME WARPING (30% prob)
speed = uniform(0.8, 1.2)  # Simula velocidad variable

# 3. RANDOM START OFFSET (30% prob)
offset = randint(0, 20)  # Simula inicio tardío
```

#### `_apply_feature_augmentation(seq)`
- Ruido gaussiano (50%)
- Frame dropout (20%)

### Configuración de datasets
```python
train_dataset = LSMDataset(..., augment=True)   # ✅
val_dataset = LSMDataset(..., augment=False)    # ✅ Puro
```

---

## 7. 🔬 Scripts de Análisis Temporal

**Archivos nuevos:**
- `src/analysis/temporal_analysis.py`
- `src/analysis/live_temporal_demo.py`

### `temporal_analysis.py`
Script forense para diagnosticar el comportamiento del buffer:
- Procesa 10 videos aleatorios
- Genera gráficas PNG por video
- Muestra estadísticas (max prob, avg prob, frames sobre umbral)

### `live_temporal_demo.py`
Demo visual que muestra video + gráfica construyéndose en tiempo real:
- Layout horizontal: video izquierda, gráfica derecha
- Colores por clase
- Umbral visualizado

---

## 8. 📦 MLflow: Logging de Código Fuente

**Archivo:** `src/training/train.py`

### Característica Nueva
Ahora se guardan los archivos de código como artifacts en cada run:

```python
code_files = [
    Path(__file__),  # train.py
    Path(__file__).parent.parent / "models" / "transformer.py",
    Path(__file__).parent.parent / "config" / "settings.py",
]
for code_file in code_files:
    mlflow.log_artifact(str(code_file), artifact_path="source_code")
```

Visible en: MLflow UI → Run → Artifacts → `source_code/`

---

## 9. 🐛 Fixes Menores

### Import `CLASSES` → `CLASS_NAMES`
**Archivos afectados:**
- `src/inference/predictor.py`
- `src/utils/file_manager.py`

El nombre de la constante cambió en `settings.py` y varios archivos tenían el import antiguo.

### Run ID en Checkpoint
**Archivo:** `src/training/train.py`

Ahora el checkpoint incluye el `run_id` de MLflow:
```python
run_id = mlflow.active_run().info.run_id

torch.save({
    ...
    'run_id': run_id
}, best_model_path)
```

Visible en demos de inferencia al cargar el modelo.

---

## 📁 Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src/inference/ipad_demo.py` | FSM, veto físico, gráfica temporal, video vertical |
| `src/inference/video_demo.py` | OneEuroFilter smoothing |
| `src/inference/predictor.py` | Fix imports |
| `src/training/train.py` | Temporal augmentation, MLflow artifacts |
| `src/utils/file_manager.py` | Fix imports |
| `src/config/settings.py` | CLASS_WEIGHTS, FEATURE_WEIGHTS, nuevas constantes |

## 📁 Archivos Nuevos

| Archivo | Propósito |
|---------|-----------|
| `src/analysis/temporal_analysis.py` | Análisis forense del buffer |
| `src/analysis/live_temporal_demo.py` | Demo visual con gráfica |
| `src/analysis/__init__.py` | Init del módulo |

---

## 🎯 Resultado Final

El sistema ahora:
1. ✅ Detecta palabras de forma estable (FSM)
2. ✅ Elimina fantasma del buffer inmediatamente (veto físico)
3. ✅ Permite señas a la altura del ombligo (margen 8%)
4. ✅ Visualiza probabilidades en tiempo real
5. ✅ Entrena con variedad temporal (evita bias de reposo)
6. ✅ Guarda código para reproducibilidad

---

## 🚀 Comandos de Uso

```bash
# Entrenar con temporal augmentation
python -m src.training.train

# Demo con FSM + veto + gráfica
python -m src.inference.ipad_demo

# Análisis forense de videos
python -m src.analysis.temporal_analysis

# Demo visual con gráfica en tiempo real
python -m src.analysis.live_temporal_demo
```
